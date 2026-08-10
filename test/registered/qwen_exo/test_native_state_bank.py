import asyncio

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.native_state_bank import (
    NativeStateBankManager,
    _apply_rotary_key,
    _atomic_torch_save,
    _dequantize_fp8,
    _load_page_payload,
    _node_mamba_value,
    _page_path,
    _quantize_fp8,
)
from qwen_exo_booster.tensor_bank import TensorBank, TensorBankCompileError


def test_native_bank_reads_unified_radix_mamba_component():
    legacy = SimpleNamespace(mamba_value=torch.tensor([3]))
    unified = SimpleNamespace(
        component_data={2: SimpleNamespace(value=torch.tensor([7]))}
    )

    assert tuple(_node_mamba_value(legacy).tolist()) == (3,)
    assert tuple(_node_mamba_value(unified).tolist()) == (7,)


class _KVPool:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.set_calls = []

    def get_key_buffer(self, _layer_id):
        return self.key

    def get_value_buffer(self, _layer_id):
        return self.value

    def set_kv_buffer(self, layer, loc, key, value, *, k_scale=None, v_scale=None):
        self.set_calls.append((layer, loc.clone(), k_scale, v_scale))
        self.key[loc] = key.to(self.key.dtype)
        self.value[loc] = value.to(self.value.dtype)


class _MambaPool:
    def __init__(self, conv, temporal):
        self.conv = conv
        self.temporal = temporal

    @staticmethod
    def translate_index(index):
        return index

    def get_cpu_copy(self, _indices):
        return [self.conv.clone()], self.temporal.clone()


def _rotary(rows=32, head_dim=8):
    half = head_dim // 2
    positions = torch.arange(rows, dtype=torch.float32).reshape(-1, 1)
    frequencies = torch.linspace(0.01, 0.08, half).reshape(1, -1)
    angles = positions * frequencies
    return SimpleNamespace(
        rotary_dim=head_dim,
        is_neox_style=True,
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1),
    )


def test_native_bank_exports_raw_kv_and_complete_section_delta(tmp_path):
    torch.manual_seed(17)
    rotary = _rotary()
    raw_key = torch.randn(6, 2, 8, dtype=torch.bfloat16)
    positions = torch.arange(6, dtype=torch.long)
    rotated_key = _apply_rotary_key(raw_key, positions=positions, rotary=rotary)
    values = torch.randn(6, 2, 8, dtype=torch.bfloat16)
    key_buffer = torch.zeros(16, 2, 8, dtype=torch.bfloat16)
    value_buffer = torch.zeros_like(key_buffer)
    physical = torch.arange(1, 7, dtype=torch.long)
    key_buffer.index_copy_(0, physical, rotated_key)
    value_buffer.index_copy_(0, physical, values)

    conv = torch.randn(1, 1, 12, 4, dtype=torch.bfloat16)
    temporal = torch.randn(1, 1, 2, 8, 8, dtype=torch.bfloat16)
    req_to_token = torch.zeros(2, 16, dtype=torch.long)
    req_to_token[1, :6] = physical
    req_pool = SimpleNamespace(
        req_to_token=req_to_token,
        mamba_pool=_MambaPool(conv, temporal),
        translate_mamba_indices=lambda indices: indices,
    )
    text_config = SimpleNamespace(
        layers_block_type=["attention"],
        model_type="qwen3_5_text",
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model_config = SimpleNamespace(model_path="", hf_text_config=text_config)
    model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(rotary_emb=rotary)])
        )
    )
    manager = NativeStateBankManager(
        root=tmp_path,
        model_config=model_config,
        model=model,
        kv_pool=_KVPool(key_buffer, value_buffer),
        kv_allocator=object(),
        req_pool=req_pool,
        tree_cache=object(),
        rank=0,
        world_size=1,
        page_size=64,
        consensus=lambda value: value,
    )
    request = SimpleNamespace(
        origin_input_ids=list(range(6)),
        req_pool_idx=1,
        mamba_pool_idx=torch.tensor(1),
    )
    manager._export(
        request,
        {
            "source_digest": "a" * 64,
            "page_id": 3,
            "capture_start": 2,
            "capture_count": 4,
            "token_start": 512,
            "prefix_identity": "page-identity",
        },
    )

    payload = _load_page_payload(tmp_path, source_digest="a" * 64, page_id=3, rank=0)
    assert payload["token_ids"] == (2, 3, 4, 5)
    manager._validate_restore_payload(
        payload, local_positions=(0, 3), prefix_token_ids=(2, 5)
    )
    with pytest.raises(RuntimeError, match="tokens do not match"):
        manager._validate_restore_payload(
            payload, local_positions=(0, 3), prefix_token_ids=(2, 4)
        )
    stored_raw = _dequantize_fp8(
        payload["full_attention"]["0"]["key"], dtype=torch.float32
    )
    rerotated = _apply_rotary_key(
        stored_raw,
        positions=torch.arange(2, 6, dtype=torch.long),
        rotary=rotary,
    )
    assert torch.allclose(
        rerotated.float(), rotated_key[2:6].float(), atol=0.13, rtol=0.08
    )
    stored_value = _dequantize_fp8(
        payload["full_attention"]["0"]["value"], dtype=torch.float32
    )
    assert torch.allclose(stored_value, values[2:6].float(), atol=0.13, rtol=0.08)

    delta = payload["section_delta"]
    stored_conv = _dequantize_fp8(delta["conv"][0], dtype=torch.float32)
    stored_temporal = _dequantize_fp8(delta["temporal"], dtype=torch.float32)
    assert stored_conv.shape == conv.shape
    assert stored_temporal.shape == temporal.shape
    assert torch.allclose(stored_conv, conv.float(), atol=0.13, rtol=0.08)
    assert torch.allclose(stored_temporal, temporal.float(), atol=0.13, rtol=0.08)


def test_rope_rebases_raw_bank_keys_to_virtual_positions():
    torch.manual_seed(19)
    rotary = _rotary(rows=128)
    raw_key = torch.randn(64, 2, 8, dtype=torch.bfloat16)
    source_positions = torch.arange(32, 96, dtype=torch.long)
    virtual_positions = torch.arange(64, dtype=torch.long)

    source_rotated = _apply_rotary_key(
        raw_key, positions=source_positions, rotary=rotary
    )
    virtual_rotated = _apply_rotary_key(
        raw_key, positions=virtual_positions, rotary=rotary
    )

    assert source_rotated.shape == virtual_rotated.shape
    assert not torch.allclose(source_rotated.float(), virtual_rotated.float())
    assert torch.allclose(
        virtual_rotated[0].float(), raw_key[0].float(), atol=0.02, rtol=0.02
    )


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [(sum(piece.encode("utf-8")) % 251) + 1 for piece in str(text).split()]

    def decode(self, token_ids, **_kwargs):
        return "token-" + "-".join(str(int(token)) for token in token_ids)


class _BankBuildRunner:
    max_fanout = 8

    def __init__(
        self,
        root: Path,
        model_fingerprint: str,
        surprisal_positions: tuple[int, ...] = (100, 300, 500),
    ):
        self.root = root
        self.model_fingerprint = model_fingerprint
        self.surprisal_positions = frozenset(surprisal_positions)
        self.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace(tp_size=1))
        self.calls = 0
        self.batch_prompt_tokens: list[tuple[int, ...]] = []

    async def run_score_batch(
        self,
        jobs,
        prompts,
        label_starts,
        sampling_params,
        *,
        custom_params_per_job,
        extra_keys,
    ):
        del sampling_params, extra_keys
        self.calls += 1
        self.batch_prompt_tokens.append(tuple(len(prompt) for prompt in prompts))
        results = []
        for job, prompt, custom in zip(jobs, prompts, custom_params_per_job):
            export = custom["qwen_exo_native_bank_export"]
            count = int(export["capture_count"])
            assert len(prompt) == int(export["capture_start"]) + count
            assert int(label_starts[len(results)]) == int(export["capture_start"]) - 1
            key = torch.zeros(count, 1, 32, dtype=torch.float32)
            if int(export["page_id"]) == 0:
                key[: min(128, count), 0, 0] = 1
                if count > 128:
                    key[128:, 0, 1] = 1
            else:
                key[:, 0, 1] = 1
            value = torch.zeros_like(key)
            payload = {
                "schema": "qwen-exo-native-state-bank-v1",
                "source_digest": export["source_digest"],
                "page_id": export["page_id"],
                "rank": 0,
                "world_size": 1,
                "model_fingerprint": self.model_fingerprint,
                "prefix_identity": export["prefix_identity"],
                "token_start": export["token_start"],
                "token_end": export["token_start"] + count,
                "capture_count": count,
                "token_ids": tuple(prompt[int(export["capture_start"]) :]),
                "full_layer_ids": (0,),
                "full_attention": {
                    "0": {
                        "key": _quantize_fp8(key, reduce_dims=(0, 2)),
                        "value": _quantize_fp8(value, reduce_dims=(0, 2)),
                    }
                },
                "section_delta": {
                    "conv": (_quantize_fp8(torch.zeros(1, 1, 2, 2), reduce_dims=(3,)),),
                    "temporal": _quantize_fp8(
                        torch.zeros(1, 1, 1, 2, 2), reduce_dims=(3, 4)
                    ),
                },
            }
            _atomic_torch_save(
                payload,
                _page_path(
                    self.root,
                    export["source_digest"],
                    export["page_id"],
                    0,
                ),
            )
            token_logprobs = tuple(
                -7.0 if position in self.surprisal_positions else -1.0
                for position in range(count)
            )
            results.append(
                SimpleNamespace(
                    job=job,
                    token_logprobs=token_logprobs,
                    metadata={"qwen_exo_bank_export_status": ["exported"]},
                )
            )
        return tuple(results)

    async def finish_parent(self, _parent_id):
        return None


class _DelayedTPBankBuildRunner(_BankBuildRunner):
    def __init__(self, root: Path, model_fingerprint: str):
        super().__init__(root, model_fingerprint)
        self.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace(tp_size=2))
        self.pending_rank_exports: list[asyncio.Task[None]] = []

    async def run_score_batch(
        self,
        jobs,
        prompts,
        label_starts,
        sampling_params,
        *,
        custom_params_per_job,
        extra_keys,
    ):
        job_list = tuple(jobs)
        prompt_list = tuple(prompts)
        custom_params = tuple(custom_params_per_job)
        results = await super().run_score_batch(
            job_list,
            prompt_list,
            label_starts,
            sampling_params,
            custom_params_per_job=custom_params,
            extra_keys=extra_keys,
        )
        for prompt, custom in zip(prompt_list, custom_params):
            export = custom["qwen_exo_native_bank_export"]
            rank_zero_path = _page_path(
                self.root,
                export["source_digest"],
                export["page_id"],
                0,
            )
            rank_zero = torch.load(rank_zero_path, weights_only=True)
            rank_zero["world_size"] = 2
            _atomic_torch_save(rank_zero, rank_zero_path)
            rank_one = dict(rank_zero)
            rank_one["rank"] = 1
            self.pending_rank_exports.append(
                asyncio.create_task(
                    self._write_rank_one(
                        rank_one,
                        _page_path(
                            self.root,
                            export["source_digest"],
                            export["page_id"],
                            1,
                        ),
                    )
                )
            )
        return results

    @staticmethod
    async def _write_rank_one(payload, path):
        await asyncio.sleep(0.02)
        _atomic_torch_save(payload, path)


@pytest.mark.asyncio
async def test_tensor_bank_waits_for_all_tp_rank_artifacts_before_loading(tmp_path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "reference.md").write_text(
        " ".join(f"token-{index}" for index in range(80)), encoding="utf-8"
    )
    repository = KnowledgeRepository(knowledge_root)
    repository.refresh()
    runner = _DelayedTPBankBuildRunner(tmp_path / "native-bank", "model-fingerprint")
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=192,
        salient_token_budget=64,
        timeout_seconds=1,
        tp_size=2,
    )

    snapshot = await bank.ensure_ready()

    assert snapshot.ready
    assert len(snapshot.pages) == 1
    assert all(task.done() for task in runner.pending_rank_exports)


@pytest.mark.asyncio
async def test_tensor_bank_batches_documents_within_fanout_and_token_budget(tmp_path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    for index in range(3):
        (knowledge_root / f"reference-{index}.md").write_text(
            " ".join(f"document-{index}-token-{token}" for token in range(80)),
            encoding="utf-8",
        )
    repository = KnowledgeRepository(knowledge_root)
    repository.refresh()
    runner = _BankBuildRunner(tmp_path / "native-bank", "model-fingerprint")
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=192,
        salient_token_budget=64,
        tp_size=1,
    )

    snapshot = await bank.ensure_ready()

    assert snapshot.ready
    assert len(snapshot.pages) == 3
    assert runner.calls == 1
    assert len(runner.batch_prompt_tokens) == 1
    assert len(runner.batch_prompt_tokens[0]) == 3
    assert sum(runner.batch_prompt_tokens[0]) <= 192 * runner.max_fanout


@pytest.mark.asyncio
async def test_tensor_bank_builds_one_document_state_with_aligned_surprisal_spans(
    tmp_path,
):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "reference.md").write_text(
        " ".join(f"token-{index}" for index in range(700)), encoding="utf-8"
    )
    repository = KnowledgeRepository(knowledge_root)
    repository.refresh()
    runner = _BankBuildRunner(tmp_path / "native-bank", "model-fingerprint")
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        tp_size=1,
        span_tokens=32,
    )

    snapshot = await bank.ensure_ready()
    assert snapshot.ready
    assert len(snapshot.pages) == 1
    assert snapshot.pages[0].token_end == 700
    assert snapshot.public_dict()["complete_gdn_document_states"] == 1
    build_calls = runner.calls
    assert await bank.ensure_ready() is snapshot
    assert runner.calls == build_calls

    query = (1.0,) + (0.0,) * 31
    (candidate,) = bank.rank(((query,),), query_identity="query", limit=1)
    selection = candidate.native_prefix
    assert selection is not None
    assert len(selection.token_ids) % 64 == 0
    assert len(selection.token_ids) >= 128
    assert {100, 300, 500}.issubset(selection.local_positions)
    assert selection.source_positions == selection.local_positions
    (other_candidate,) = bank.rank(
        (((0.0, 1.0) + (0.0,) * 30,),),
        query_identity="query-other",
        limit=1,
    )
    assert other_candidate.native_prefix is not None
    assert other_candidate.native_prefix.prefix_identity != selection.prefix_identity
    assert other_candidate.native_prefix.local_positions != selection.local_positions
    assert candidate.virtual_positions == tuple(range(len(selection.source_positions)))
    assert selection.scheduler_payload() == {
        "source_digest": snapshot.source_digest,
        "page_id": 0,
        "local_positions": list(selection.local_positions),
        "prefix_identity": selection.prefix_identity,
    }

    calls = runner.calls
    reloaded = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        tp_size=1,
        span_tokens=32,
    )
    assert (await reloaded.ensure_ready()).ready
    assert runner.calls == calls
    assert len(selection.token_ids) <= bank.salient_token_budget
    query_positions = bank._query_conditioned_positions(
        snapshot.pages[0],
        state_token_count=700,
        query_anchor_positions=(200,),
    )
    assert set(range(168, 232)).issubset(query_positions)

    bounded_positions = bank._query_conditioned_positions(
        snapshot.pages[0],
        state_token_count=431,
        query_anchor_positions=(100, 180, 260, 340, 420, 500, 580, 660),
    )
    assert len(bounded_positions) <= 431
    assert len(bounded_positions) % 64 == 0


@pytest.mark.asyncio
async def test_tensor_bank_rejects_document_when_merged_spans_exceed_budget(tmp_path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "too-dense.md").write_text(
        " ".join(f"token-{index}" for index in range(256)), encoding="utf-8"
    )
    repository = KnowledgeRepository(knowledge_root)
    repository.refresh()
    runner = _BankBuildRunner(
        tmp_path / "native-bank",
        "model-fingerprint",
        surprisal_positions=tuple(range(0, 256, 32)),
    )
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=256,
        salient_token_budget=64,
        surprisal_threshold=6.0,
        span_tokens=32,
        tp_size=1,
    )

    with pytest.raises(TensorBankCompileError) as captured:
        await bank.ensure_ready()

    error = captured.value
    assert error.code == "salient_span_budget_exceeded"
    assert error.relative_path == "too-dense.md"
    assert error.details["salient_tokens"] > 64
    assert error.details["salient_token_budget"] == 64
    assert "Split or simplify" in error.hint
    assert not (tmp_path / "tensor-bank.pt").exists()
    assert not (tmp_path / "native-bank" / bank._failure_digest).exists()
    calls = runner.calls
    with pytest.raises(TensorBankCompileError):
        await bank.ensure_ready()
    assert runner.calls == calls


@pytest.mark.asyncio
async def test_tensor_bank_rejects_source_length_before_scheduler_dispatch(tmp_path):
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    (knowledge_root / "too-long.md").write_text(
        " ".join(f"token-{index}" for index in range(65)), encoding="utf-8"
    )
    repository = KnowledgeRepository(knowledge_root)
    repository.refresh()
    runner = _BankBuildRunner(tmp_path / "native-bank", "model-fingerprint")
    bank = TensorBank(
        tmp_path / "tensor-bank.pt",
        runner,
        _WordTokenizer(),
        {"knowledge": repository},
        model_fingerprint="model-fingerprint",
        max_document_tokens=64,
        salient_token_budget=64,
        tp_size=1,
    )

    with pytest.raises(TensorBankCompileError) as captured:
        await bank.ensure_ready()

    assert captured.value.code == "document_token_limit_exceeded"
    assert captured.value.details == {
        "source_tokens": 65,
        "max_document_tokens": 64,
    }
    assert "64-token salient budget is not a source-length limit" in captured.value.hint
    assert runner.calls == 0


class _SequentialAllocator:
    def __init__(self, start):
        self.start = start
        self.freed = []

    def alloc(self, count):
        return torch.arange(self.start, self.start + count, dtype=torch.long)

    def free(self, indices):
        self.freed.append(tuple(int(item) for item in indices.reshape(-1)))


class _LoadMambaPool:
    def __init__(self):
        self.loaded = None

    @staticmethod
    def translate_index(index):
        return index

    def load_cpu_copy(self, state, indices):
        self.loaded = (state, indices.clone())


class _InsertCache:
    def __init__(self):
        self.inserted = None

    def insert(self, params):
        self.inserted = (
            params.key,
            params.value.clone(),
            params.mamba_value.clone(),
        )
        return SimpleNamespace(prefix_len=0, mamba_exist=False)

    def evict(self, _params):
        raise AssertionError("restore unexpectedly needed eviction")


def test_native_bank_restore_materializes_selected_kv_and_section_delta(tmp_path):
    torch.manual_seed(23)
    rotary = _rotary(rows=256)
    raw_key = torch.randn(192, 2, 8, dtype=torch.float32)
    value = torch.randn(192, 2, 8, dtype=torch.float32)
    conv = torch.randn(1, 1, 12, 4, dtype=torch.float32)
    temporal = torch.randn(1, 1, 2, 8, 8, dtype=torch.float32)
    payload = {
        "full_attention": {
            "0": {
                "key": _quantize_fp8(raw_key, reduce_dims=(0, 2)),
                "value": _quantize_fp8(value, reduce_dims=(0, 2)),
            }
        },
        "section_delta": {
            "conv": (_quantize_fp8(conv, reduce_dims=(3,)),),
            "temporal": _quantize_fp8(temporal, reduce_dims=(3, 4)),
        },
    }
    key_buffer = torch.zeros(256, 2, 8, dtype=torch.bfloat16)
    value_buffer = torch.zeros_like(key_buffer)
    kv_allocator = _SequentialAllocator(10)
    mamba_allocator = _SequentialAllocator(7)
    mamba_pool = _LoadMambaPool()
    req_pool = SimpleNamespace(
        mamba_allocator=mamba_allocator,
        mamba_pool=mamba_pool,
        translate_mamba_indices=lambda indices: indices,
    )
    tree_cache = _InsertCache()
    text_config = SimpleNamespace(
        layers_block_type=["attention"],
        model_type="qwen3_5_text",
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    attention = SimpleNamespace(
        layer_id=0, k_scale=torch.tensor(1.0), v_scale=torch.tensor(1.0)
    )
    kv_pool = _KVPool(key_buffer, value_buffer)
    manager = NativeStateBankManager(
        root=tmp_path,
        model_config=SimpleNamespace(model_path="", hf_text_config=text_config),
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[SimpleNamespace(rotary_emb=rotary, attn=attention)]
                )
            )
        ),
        kv_pool=kv_pool,
        kv_allocator=kv_allocator,
        req_pool=req_pool,
        tree_cache=tree_cache,
        rank=0,
        world_size=1,
        page_size=64,
        consensus=lambda value: value,
        insert_params_factory=lambda **values: SimpleNamespace(**values),
    )
    selected = tuple(range(0, 192, 3))
    manager._restore_prefix(
        SimpleNamespace(),
        payload=payload,
        key="native-key",
        local_positions=selected,
    )
    assert len(kv_pool.set_calls) == 1
    called_attention, called_locations, called_k_scale, called_v_scale = (
        kv_pool.set_calls[0]
    )
    assert called_attention is attention
    assert tuple(called_locations.tolist()) == tuple(range(10, 74))
    assert called_k_scale is attention.k_scale
    assert called_v_scale is attention.v_scale

    physical = torch.arange(10, 74, dtype=torch.long)
    expected_key = _apply_rotary_key(
        raw_key.index_select(0, torch.tensor(selected)),
        positions=torch.arange(64),
        rotary=rotary,
    )
    assert torch.allclose(
        key_buffer.index_select(0, physical).float(),
        expected_key.float(),
        atol=0.14,
        rtol=0.08,
    )
    assert torch.allclose(
        value_buffer.index_select(0, physical).float(),
        value.index_select(0, torch.tensor(selected)),
        atol=0.14,
        rtol=0.08,
    )
    assert tree_cache.inserted[0] == "native-key"
    assert tuple(tree_cache.inserted[1].tolist()) == tuple(physical.tolist())
    assert tuple(tree_cache.inserted[2].tolist()) == (7,)
    assert mamba_pool.loaded is not None
    loaded_conv, loaded_temporal = mamba_pool.loaded[0]
    assert torch.allclose(loaded_conv[0].float(), conv, atol=0.14, rtol=0.08)
    assert torch.allclose(loaded_temporal.float(), temporal, atol=0.14, rtol=0.08)
