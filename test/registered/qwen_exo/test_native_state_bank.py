import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.native_state_bank import (NativeStateBankManager, NativeStateBankError, _atomic_torch_save, _session_initial_gdn_path, validate_session_initial_gdn_artifacts)


class _KVPool:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.set_calls = []

    def get_key_buffer(self, _layer_id):
        return self.key

    def get_value_buffer(self, _layer_id):
        return self.value

    def get_kv_tokens(self, _layer_id, indices, dtype):
        return (
            self.key.index_select(0, indices).to(dtype),
            self.value.index_select(0, indices).to(dtype),
        )

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


class _SessionMambaPool:
    def __init__(self, *, replay=False, ple=False):
        self.mamba_cache = SimpleNamespace(
            conv=[torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)],
            temporal=torch.zeros(1, 4, 1, 2, 2, dtype=torch.bfloat16),
        )
        self.replayssm_write_pos = torch.zeros(4, dtype=torch.int32) if replay else None
        self.replayssm_cache_base = torch.zeros(4, dtype=torch.int32) if replay else None
        self.replayssm_is_flush = torch.zeros(4, dtype=torch.int8) if replay else None
        self._slot_siblings = (
            (
                SimpleNamespace(conv_state=torch.zeros(2, 4, 2, 3, dtype=torch.bfloat16)),
                SimpleNamespace(context=torch.zeros(4, 2, dtype=torch.int64)),
            )
            if ple
            else ()
        )

    def get_cpu_copy(self, indices):
        indices = indices.to(dtype=torch.long)
        state = [
            [self.mamba_cache.conv[0][:, indices].cpu().clone()],
            self.mamba_cache.temporal[:, indices].cpu().clone(),
        ]
        if self.replayssm_cache_base is not None:
            state.append(
                [cursor[indices].clone() for cursor in (
                    self.replayssm_write_pos, self.replayssm_cache_base, self.replayssm_is_flush
                )]
            )
        if self._slot_siblings:
            state.append([
                self._slot_siblings[0].conv_state[:, indices].clone(),
                self._slot_siblings[1].context[indices].clone(),
            ])
        return tuple(state)

    def load_cpu_copy(self, state, indices):
        indices = indices.to(dtype=torch.long)
        conv, temporal = state[:2]
        self.mamba_cache.conv[0][:, indices] = conv[0]
        self.mamba_cache.temporal[:, indices] = temporal
        if self.replayssm_cache_base is not None:
            for target, source in zip(
                (self.replayssm_write_pos, self.replayssm_cache_base, self.replayssm_is_flush),
                state[2],
            ):
                target[indices] = source
        if self._slot_siblings:
            self._slot_siblings[0].conv_state[:, indices] = state[-1][0]
            self._slot_siblings[1].context[indices] = state[-1][1]


class _SessionMambaAllocator:
    def __init__(self):
        self.allocations = 0

    def alloc(self, count):
        assert count == 1
        self.allocations += 1
        return torch.tensor([1 + self.allocations], dtype=torch.int32)

    def free(self, _indices):
        return None


def _session_artifact(state, rank=0):
    return {
        "schema": "qwen-exo-session-initial-gdn-v1",
        "source_digest": "a" * 64,
        "state_identity": "b" * 64,
        "rank": rank,
        "world_size": 1,
        "model_fingerprint": "session-model",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "mamba_state": state,
    }


def _validate_public_session_artifact(tmp_path, payload):
    _atomic_torch_save(
        payload,
        _session_initial_gdn_path(
            tmp_path, source_digest="a" * 64, state_identity="b" * 64, rank=0
        ),
    )
    return validate_session_initial_gdn_artifacts(
        tmp_path,
        source_digest="a" * 64,
        state_identity="b" * 64,
        world_size=1,
        model_fingerprint="session-model",
    )


def _session_state_layout(*, replay=True, ple=True):
    state = [
        [torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)],
        torch.zeros(1, 1, 1, 2, 2, dtype=torch.bfloat16),
    ]
    if replay:
        state.append(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int8),
            ]
        )
    if ple:
        state.append(
            [
                torch.zeros(2, 1, 2, 3, dtype=torch.bfloat16),
                torch.tensor([[11, 12]], dtype=torch.int64),
            ]
        )
    return state


def _session_validator_manager(*, replay=True, ple=True):
    pool = _SessionMambaPool(replay=replay, ple=ple)
    manager = object.__new__(NativeStateBankManager)
    manager.req_pool = SimpleNamespace(mamba_pool=pool)
    manager.world_size = 1
    manager.model_fingerprint = "session-model"
    return manager


def _export_session_artifact(tmp_path):
    manager = _session_validator_manager()
    manager.root = tmp_path
    manager.rank = 0
    manager.req_pool.translate_mamba_indices = lambda indices: indices
    manager._export_session_initial_gdn(
        SimpleNamespace(
            mamba_pool_idx=torch.tensor(1),
            origin_input_ids=list(range(7)),
            output_ids_through_stop=[8, 9, 10],
        ),
        {"source_digest": "a" * 64, "state_identity": "b" * 64},
    )
    return torch.load(
        _session_initial_gdn_path(
            tmp_path, source_digest="a" * 64, state_identity="b" * 64, rank=0
        ),
        map_location="cpu",
        weights_only=True,
    )


@pytest.mark.parametrize("replay,ple", [(False, False), (True, False), (False, True), (True, True)])
def test_session_artifact_accepts_dense_replay_and_ple_layouts(tmp_path, replay, ple):
    payload = _session_artifact(_session_state_layout(replay=replay, ple=ple))
    assert _validate_public_session_artifact(tmp_path, payload) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }
    _session_validator_manager(replay=replay, ple=ple)._validate_session_initial_gdn_payload(payload)


@pytest.mark.parametrize("sibling_index", [0, 1])
def test_session_artifact_accepts_one_enabled_ple_sibling(tmp_path, sibling_index):
    state = _session_state_layout(replay=False)
    state[2] = [state[2][sibling_index]]
    payload = _session_artifact(state)
    assert _validate_public_session_artifact(tmp_path, payload) == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }
    manager = _session_validator_manager(replay=False)
    manager.req_pool.mamba_pool._slot_siblings = (
        manager.req_pool.mamba_pool._slot_siblings[sibling_index],
    )
    manager._validate_session_initial_gdn_payload(payload)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("conv", []),
        ("conv", [torch.zeros(1, 2, 2, 2)]),
        ("conv", [torch.zeros(2, 1, 2, 2)]),
        ("conv", [torch.zeros(1, 1, 2, 2, dtype=torch.int64)]),
        ("conv", [torch.full((1, 1, 2, 2), float("nan"))]),
        ("temporal", None),
        ("temporal", torch.zeros(1, 2, 1, 2, 2)),
        ("temporal", torch.full((1, 1, 1, 2, 2), float("inf"))),
        ("cursors", None),
        ("cursors", [torch.zeros(1, dtype=torch.int32)]),
        ("cursor_0", torch.ones(1, dtype=torch.int32)),
        ("cursor_1", torch.ones(1, dtype=torch.int32)),
        ("cursor_2", torch.ones(1, dtype=torch.int8)),
        ("cursor_0", torch.zeros(1, dtype=torch.float32)),
        ("cursor_1", torch.zeros(1, dtype=torch.int64)),
        ("cursor_2", torch.zeros(1, dtype=torch.int32)),
        ("cursor_0", torch.zeros(1, 1, dtype=torch.int32)),
        ("cursor_0", torch.zeros(2, dtype=torch.int32)),
        ("siblings", None),
        ("siblings", []),
        ("siblings", [torch.zeros(2, 2)]),
        ("sibling_0", torch.zeros(2, 2, 2, 3)),
        ("sibling_0", torch.zeros(2, 1, 2, 3, dtype=torch.int64)),
        ("sibling_0", torch.full((2, 1, 2, 3), float("nan"))),
        ("sibling_1", torch.zeros(2, 2, dtype=torch.int64)),
        ("sibling_1", torch.zeros(1, 2, dtype=torch.float32)),
    ],
)
def test_session_artifact_rejects_malformed_four_part_state(tmp_path, field, replacement):
    payload = _export_session_artifact(tmp_path)
    state = list(payload["mamba_state"])
    if field.startswith("cursor_"):
        state[2][int(field[-1])] = replacement
    elif field.startswith("sibling_"):
        state[3][int(field[-1])] = replacement
    else:
        state[{"conv": 0, "temporal": 1, "cursors": 2, "siblings": 3}[field]] = replacement
    payload["mamba_state"] = state
    with pytest.raises(NativeStateBankError):
        _validate_public_session_artifact(tmp_path, payload)
    with pytest.raises(NativeStateBankError):
        _session_validator_manager()._validate_session_initial_gdn_payload(payload)


@pytest.mark.parametrize(
    "components",
    [
        None,
        ("conv", "temporal", "slot_siblings", "replayssm_cursors"),
        ("conv", "temporal", "replayssm_cursors", "unknown"),
        ("conv", "temporal", "replayssm_cursors"),
    ],
)
def test_session_artifact_rejects_invalid_component_declaration(tmp_path, components):
    payload = _export_session_artifact(tmp_path)
    payload["mamba_state_components"] = components
    with pytest.raises(NativeStateBankError, match="declared component layout"):
        _validate_public_session_artifact(tmp_path, payload)


@pytest.mark.parametrize("third_part", [2, 3])
def test_session_artifact_requires_declared_cursor_or_sibling_bundle_type(tmp_path, third_part):
    payload = _export_session_artifact(tmp_path)
    state = payload["mamba_state"]
    payload["mamba_state"] = (state[0], state[1], state[third_part])
    payload["mamba_state_components"] = (
        "conv", "temporal",
        "slot_siblings" if third_part == 2 else "replayssm_cursors",
    )
    with pytest.raises(NativeStateBankError):
        _validate_public_session_artifact(tmp_path, payload)


@pytest.mark.parametrize("missing", ["cursors", "siblings", "both"])
def test_session_adoption_rejects_missing_required_replay_or_ple_state(tmp_path, missing):
    payload = _export_session_artifact(tmp_path)
    state = list(payload["mamba_state"])
    if missing == "cursors":
        del state[2]
    elif missing == "siblings":
        del state[3]
    else:
        del state[2:]
    payload["mamba_state"] = state
    with pytest.raises(NativeStateBankError, match="declared component layout"):
        _validate_public_session_artifact(tmp_path, payload)
    with pytest.raises(NativeStateBankError, match="declared component layout"):
        _session_validator_manager()._validate_session_initial_gdn_payload(payload)


@pytest.mark.parametrize("replay,ple", [(False, False), (True, True)])
def test_session_initial_gdn_refresh_keeps_previous_identity_bindable(tmp_path, replay, ple):
    """A refreshed identity must not reject requests queued under the old one.

    The scheduler prepares the source slot when a request arrives and binds it
    when the request is scheduled. With a single reserved slot, a newer
    identity arriving in between overwrote the slot and the older request was
    rejected at bind time. The bank keeps the previous identity resident and
    only recycles the least recently used slot for a third identity.
    """
    mamba_pool = _SessionMambaPool(replay=replay, ple=ple)
    allocator = _SessionMambaAllocator()
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool,
        mamba_allocator=allocator,
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
            language_model=SimpleNamespace(layers=[SimpleNamespace()])
        )
    )
    manager = NativeStateBankManager(
        root=tmp_path,
        model_config=model_config,
        model=model,
        kv_pool=object(),
        kv_allocator=object(),
        req_pool=req_pool,
        tree_cache=SimpleNamespace(evict=lambda _params: None),
        rank=0,
        world_size=1,
        page_size=64,
        consensus=lambda value: value,
    )

    def export(identity, value):
        mamba_pool.mamba_cache.conv[0][:, 1].fill_(value)
        mamba_pool.mamba_cache.temporal[:, 1].fill_(value)
        if ple:
            mamba_pool._slot_siblings[0].conv_state[:, 1].fill_(value)
            mamba_pool._slot_siblings[1].context[1].fill_(int(value) + 10)
        request = SimpleNamespace(
            sampling_params=SimpleNamespace(
                custom_params={
                    "qwen_exo_session_initial_gdn_export": {
                        "source_digest": identity[0],
                        "state_identity": identity[1],
                    }
                }
            ),
            mamba_pool_idx=torch.tensor(1),
            origin_input_ids=[1, 2, 3],
            output_ids_through_stop=[4, 5],
        )
        assert manager.maybe_export(request) is True
        assert request.qwen_exo_session_initial_gdn_status == "exported"
        assert validate_session_initial_gdn_artifacts(
            tmp_path,
            source_digest=identity[0],
            state_identity=identity[1],
            world_size=1,
            model_fingerprint=manager.model_fingerprint,
        ) == {"prompt_tokens": 3, "completion_tokens": 2}

    def selection(identity):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(
                custom_params={
                    "qwen_exo_session_initial_gdn": {
                        "source_digest": identity[0],
                        "state_identity": identity[1],
                        "cache_namespace": "session-cache",
                    }
                }
            ),
            extra_key="session-cache|qwen-exo-editor=none",
            prefix_indices=(),
            mamba_cow_src_index=None,
            mamba_needs_clear=True,
        )

    first_identity = ("a" * 64, "b" * 64)
    export(first_identity, 3)
    queued_request = selection(first_identity)
    assert manager.ensure_session_initial_gdn_source(queued_request) is True
    assert manager.reserved_mamba_slots() == 1

    # A refreshed identity arrives before the queued request is scheduled.
    second_identity = ("c" * 64, "d" * 64)
    export(second_identity, 7)
    second_request = selection(second_identity)
    assert manager.ensure_session_initial_gdn_source(second_request) is True
    assert manager.reserved_mamba_slots() == 2

    assert manager.bind_session_initial_gdn(queued_request) is True
    assert tuple(queued_request.mamba_cow_src_index.tolist()) == (2,)
    assert queued_request.mamba_needs_clear is False
    assert torch.all(mamba_pool.mamba_cache.temporal[:, 2] == 3)
    assert manager.bind_session_initial_gdn(second_request) is True
    assert tuple(second_request.mamba_cow_src_index.tolist()) == (3,)
    assert torch.all(mamba_pool.mamba_cache.temporal[:, 3] == 7)
    assert allocator.allocations == 2

    # A third identity recycles the least recently used slot (the first one)
    # instead of growing the reserve; the evicted identity reloads on demand.
    third_identity = ("e" * 64, "f" * 64)
    export(third_identity, 9)
    third_request = selection(third_identity)
    assert manager.bind_session_initial_gdn(third_request) is True
    assert tuple(third_request.mamba_cow_src_index.tolist()) == (2,)
    assert torch.all(mamba_pool.mamba_cache.temporal[:, 2] == 9)
    assert allocator.allocations == 2
    assert manager.reserved_mamba_slots() == 2

    revived_request = selection(first_identity)
    assert manager.bind_session_initial_gdn(revived_request) is True
    assert tuple(revived_request.mamba_cow_src_index.tolist()) == (3,)
    assert torch.all(mamba_pool.mamba_cache.temporal[:, 3] == 3)
    if ple:
        assert torch.all(mamba_pool._slot_siblings[0].conv_state[:, 3] == 3)
        assert torch.all(mamba_pool._slot_siblings[1].context[3] == 13)
        assert torch.all(mamba_pool._slot_siblings[0].conv_state[:, 2] == 9)
        assert torch.all(mamba_pool._slot_siblings[1].context[2] == 19)
    if replay:
        assert torch.equal(mamba_pool.replayssm_cache_base[2:4], torch.zeros(2, dtype=torch.int32))
        assert torch.equal(mamba_pool.replayssm_is_flush[2:4], torch.zeros(2, dtype=torch.int8))
    assert manager.stats()["session_gdn_loads"] == 4
    assert manager.stats()["session_gdn_binds"] == 4


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

    async def finish_parent(self, _parent_id):
        return None


class _DelayedTPBankBuildRunner(_BankBuildRunner):
    def __init__(self, root: Path, model_fingerprint: str):
        super().__init__(root, model_fingerprint)
        self.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace(tp_size=2))
        self.pending_rank_exports: list[asyncio.Task[None]] = []

    @staticmethod
    async def _write_rank_one(payload, path):
        await asyncio.sleep(0.02)
        _atomic_torch_save(payload, path)


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


def test_session_initial_gdn_export_rejects_nonfinite_state_before_saving(tmp_path):
    manager = _session_validator_manager()
    manager.root = tmp_path
    manager.rank = 0
    manager.req_pool.translate_mamba_indices = lambda indices: indices
    manager.req_pool.mamba_pool.mamba_cache.temporal[:, 1].fill_(float("nan"))
    with pytest.raises(NativeStateBankError):
        manager._export_session_initial_gdn(
            SimpleNamespace(
                mamba_pool_idx=torch.tensor(1), origin_input_ids=[1, 2],
                output_ids_through_stop=[3],
            ),
            {"source_digest": "a" * 64, "state_identity": "b" * 64},
        )
    assert not _session_initial_gdn_path(
        tmp_path, source_digest="a" * 64, state_identity="b" * 64, rank=0,
    ).exists()

