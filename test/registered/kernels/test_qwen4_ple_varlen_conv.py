"""Correctness coverage for Qwen4 packed-varlen PLE short convolution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.qwen4_ple import (
    can_fuse_qwen4_varlen_conv,
    fused_qwen4_varlen_conv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import _use_qwen4_varlen_prefill
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


def _metadata(lengths, state_indices):
    lengths_tensor = torch.tensor(lengths, device="cuda", dtype=torch.long)
    query_start_loc = torch.cat([lengths_tensor.new_zeros(1), lengths_tensor.cumsum(0)])
    req_indices = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda"), lengths_tensor
    )
    token_offsets = (
        torch.arange(sum(lengths), device="cuda") - query_start_loc[req_indices]
    )
    return (
        query_start_loc,
        req_indices,
        token_offsets,
        torch.tensor(state_indices, device="cuda", dtype=torch.long),
    )


def _inputs(
    tokens,
    channels,
    slots,
    kernel_size,
    dilation,
    seed,
    *,
    state_dtype=torch.bfloat16,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    x = torch.randn(
        (tokens, channels),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (channels, 1, kernel_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    state = torch.randn(
        (slots, channels, state_len),
        device="cuda",
        dtype=state_dtype,
        generator=generator,
    )
    return x, weight, state


def _reference(x, weight, state, query_start_loc, state_indices, dilation):
    output = torch.empty_like(x)
    next_state = state.clone()
    state_len = state.shape[2]
    for req, slot in enumerate(state_indices.tolist()):
        start = int(query_start_loc[req])
        end = int(query_start_loc[req + 1])
        if start == end:
            continue
        conv_input = torch.cat(
            [state[slot].to(x.dtype), x[start:end].T], dim=1
        ).unsqueeze(0)
        conv = F.conv1d(
            conv_input.float(),
            weight.float(),
            dilation=dilation,
            groups=x.shape[1],
        ).to(x.dtype)
        output[start:end] = conv.squeeze(0).T
        if slot != 0:
            next_state[slot] = conv_input[0, :, end - start : end - start + state_len]
    return output, next_state


@requires_cuda
class TestQwen4ShortConv(CustomTestCase):
    def test_packed_varlen_matches_padded_reference(self):
        lengths = [1, 2, 8, 9, 10, 33, 0]
        state_indices = [7, 2, 9, 4, 11, 6, 0]
        x, weight, initial_state = _inputs(
            sum(lengths), 257, 12, kernel_size=3, dilation=4, seed=1
        )
        query_start_loc, req_indices, token_offsets, state_indices_tensor = _metadata(
            lengths, state_indices
        )
        expected, expected_state = _reference(
            x,
            weight,
            initial_state,
            query_start_loc,
            state_indices_tensor,
            dilation=4,
        )
        actual_state = initial_state.clone()
        actual = fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            state_indices_tensor,
            query_start_loc,
            req_indices,
            token_offsets,
            dilation=4,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_writeback_updates_main_and_track_boundaries(self):
        lengths = [12]
        x, weight, initial_state = _inputs(
            12, 129, 10, kernel_size=3, dilation=4, seed=2
        )
        query_start_loc, req_indices, token_offsets, state_indices = _metadata(
            lengths, [3]
        )
        track_indices = torch.tensor([8], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([5], device="cuda", dtype=torch.long)
        actual_state = initial_state.clone()
        fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            state_indices,
            query_start_loc,
            req_indices,
            token_offsets,
            dilation=4,
            track_indices=track_indices,
            track_offsets=track_offsets,
        )

        full_input = torch.cat([initial_state[3], x.T], dim=1)
        torch.testing.assert_close(actual_state[3], full_input[:, 12:20])
        torch.testing.assert_close(actual_state[8], full_input[:, 5:13])

    def test_chunked_writeback_matches_one_shot(self):
        x, weight, initial_state = _inputs(
            12, 129, 8, kernel_size=3, dilation=4, seed=3
        )
        one_state = initial_state.clone()
        one_meta = _metadata([12], [5])
        one_output = fused_qwen4_varlen_conv(
            x, weight, one_state, one_meta[3], *one_meta[:3], dilation=4
        )

        chunk_state = initial_state.clone()
        outputs = []
        for chunk in (x[:5], x[5:]):
            meta = _metadata([chunk.shape[0]], [5])
            outputs.append(
                fused_qwen4_varlen_conv(
                    chunk, weight, chunk_state, meta[3], *meta[:3], dilation=4
                )
            )

        torch.testing.assert_close(
            torch.cat(outputs).float(), one_output.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(chunk_state, one_state))

    def test_writeback_skips_empty_rows_and_dummy_slot_zero(self):
        x, weight, initial_state = _inputs(2, 65, 8, kernel_size=3, dilation=4, seed=4)
        meta = _metadata([2, 0, 0], [0, 4, 5])
        track_indices = torch.tensor([0, 6, 7], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([1, 0, 0], device="cuda", dtype=torch.long)
        actual_state = initial_state.clone()
        fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            meta[3],
            *meta[:3],
            dilation=4,
            track_indices=track_indices,
            track_offsets=track_offsets,
        )
        self.assertTrue(torch.equal(actual_state, initial_state))

    def test_fp32_state_matches_bf16_materialization_of_padded_path(self):
        lengths = [1, 5]
        x, weight, initial_state = _inputs(
            sum(lengths),
            65,
            6,
            kernel_size=3,
            dilation=4,
            seed=6,
            state_dtype=torch.float32,
        )
        # Make loss at the old path's FP32 -> BF16 boundary deterministic.
        initial_state.add_(0.00390625)
        meta = _metadata(lengths, [2, 5])
        expected, expected_state = _reference(
            x, weight, initial_state, meta[0], meta[3], dilation=4
        )
        actual_state = initial_state.clone()
        actual = fused_qwen4_varlen_conv(
            x, weight, actual_state, meta[3], *meta[:3], dilation=4
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_metadata_requires_int64_and_same_cuda_device(self):
        x, weight, state = _inputs(3, 65, 5, kernel_size=3, dilation=4, seed=7)
        metadata = _metadata([1, 2], [2, 4])
        args = [x, weight, state, metadata[3], *metadata[:3], 4]

        for index in range(3, 7):
            invalid = list(args)
            invalid[index] = invalid[index].to(dtype=torch.int32)
            self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))

        invalid = list(args)
        invalid[1] = invalid[1].cpu()
        self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))
        if torch.cuda.device_count() > 1:
            invalid = list(args)
            invalid[1] = invalid[1].to("cuda:1")
            self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))

    def test_track_metadata_requires_int64_and_same_device(self):
        x, weight, state = _inputs(3, 65, 8, kernel_size=3, dilation=4, seed=8)
        metadata = _metadata([3], [2])
        track_indices = torch.tensor([6], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([2], device="cuda", dtype=torch.long)

        for invalid_indices, invalid_offsets in (
            (track_indices.int(), track_offsets),
            (track_indices, track_offsets.int()),
            (track_indices.cpu(), track_offsets),
        ):
            with self.assertRaisesRegex(ValueError, "track metadata"):
                fused_qwen4_varlen_conv(
                    x,
                    weight,
                    state.clone(),
                    metadata[3],
                    *metadata[:3],
                    dilation=4,
                    track_indices=invalid_indices,
                    track_offsets=invalid_offsets,
                )
        if torch.cuda.device_count() > 1:
            with self.assertRaisesRegex(ValueError, "track metadata"):
                fused_qwen4_varlen_conv(
                    x,
                    weight,
                    state.clone(),
                    metadata[3],
                    *metadata[:3],
                    dilation=4,
                    track_indices=track_indices.to("cuda:1"),
                    track_offsets=track_offsets,
                )

    def test_dispatch_is_limited_to_ordinary_bf16_prefill(self):
        x, weight, state = _inputs(3, 65, 5, kernel_size=3, dilation=4, seed=5)
        metadata = _metadata([1, 2], [2, 4])
        kernel_args = (x, weight, state, metadata[3], *metadata[:3], 4)
        dispatch_args = (
            x,
            weight,
            state,
            metadata[3],
            metadata[1],
            metadata[2],
            4,
        )
        self.assertTrue(can_fuse_qwen4_varlen_conv(*kernel_args))
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
        ):
            self.assertTrue(
                _use_qwen4_varlen_prefill(mode, metadata[0], *dispatch_args)
            )
        self.assertFalse(
            _use_qwen4_varlen_prefill(
                ForwardMode.TARGET_VERIFY, metadata[0], *dispatch_args
            )
        )
        self.assertFalse(
            can_fuse_qwen4_varlen_conv(
                x.float(), weight, state, metadata[3], *metadata[:3], 4
            )
        )
        with qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(False):
            self.assertFalse(
                _use_qwen4_varlen_prefill(
                    ForwardMode.EXTEND, metadata[0], *dispatch_args
                )
            )
        with patch.object(torch.version, "hip", "unsupported-rocm"):
            self.assertFalse(
                _use_qwen4_varlen_prefill(
                    ForwardMode.EXTEND, metadata[0], *dispatch_args
                )
            )


    def test_real_ple_shape_and_nine_column_state(self):
        for lengths in ([33, 1, 1, 0], [9, 9, 9, 9], [0, 0, 0, 0]):
            for state_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(lengths=lengths, state_dtype=state_dtype):
                    x, weight, initial = _inputs(
                        sum(lengths), 10240, 9, 4, 3, 9,
                        state_dtype=state_dtype,
                    )
                    meta = _metadata(lengths, [1, 2, 3, 4])
                    expected, expected_state = _reference(
                        x, weight, initial, meta[0], meta[3], 3
                    )
                    state = initial.clone()
                    actual = fused_qwen4_varlen_conv(
                        x, weight, state, meta[3], *meta[:3], 3
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
                    self.assertTrue(torch.equal(state, expected_state))

    def test_model_dispatch_preserves_local_tracking_and_chunk_offsets(self):
        lengths = [12, 0, 5]
        x, weight, initial = _inputs(17, 65, 8, 4, 3, 10, state_dtype=torch.float32)
        state = initial.clone()
        pool = SimpleNamespace(
            ple_window_cache=None,
            get_mamba_indices=lambda indices: indices,
            short_conv_layer_cache=lambda layer: state,
        )
        forward = ForwardBatch(
            forward_mode=ForwardMode.EXTEND, batch_size=3,
            input_ids=torch.zeros(20, dtype=torch.long, device="cuda"),
            req_pool_indices=torch.tensor([1, 2, 3], device="cuda"),
            seq_lens=torch.tensor([44, 0, 69], device="cuda"),
            out_cache_loc=torch.arange(1, 21, device="cuda"),
            seq_lens_sum=113, return_logprob=False,
            positions=torch.arange(20, device="cuda"),
            extend_seq_lens=torch.tensor(lengths, device="cuda"),
            extend_seq_lens_cpu=lengths, num_token_non_padded_cpu=17,
            extend_prefix_lens=torch.tensor([32, 0, 64], device="cuda"),
            mamba_track_indices=torch.tensor([5, 6, 7], device="cuda"),
            mamba_track_mask=torch.tensor([True, False, True], device="cuda"),
            mamba_track_seqlens=torch.tensor([40, 0, 68], device="cuda"),
        )
        layer = SimpleNamespace(
            layer_id=1, conv1d=SimpleNamespace(weight=weight),
            short_conv_dilation=3, short_conv_state_len=9, conv_channels=65,
        )
        from sglang.srt.runtime_context import get_context

        with (
            forward_context(ForwardContext(attn_backend=SimpleNamespace(req_to_token_pool=pool))),
            get_context().override_server_args(mamba_cache_chunk_size=4),
        ):
            batch = qwen4_exp._prepare_ple_batch(
                forward.input_ids, forward, ngram_size=None, ngram_eos_token_id=None
            )
            self.assertEqual(batch.query_start_loc.tolist(), [0, 12, 12, 17])
            self.assertEqual(batch.state_indices.tolist(), [1, 0, 3])
            self.assertEqual(batch.physical_tokens, 20)
            for enabled in (False, True):
                state.copy_(initial)
                with qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(enabled):
                    actual = qwen4_exp.Qwen4ExpPLELayer._short_conv(layer, x, forward, batch)
                if not enabled:
                    expected, expected_state = actual.clone(), state.clone()
                else:
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
                    self.assertTrue(torch.equal(state[1:], expected_state[1:]))
                    self.assertTrue(torch.equal(state[0], initial[0]))
            for slot, start, end, offset in ((5, 0, 12, 8), (7, 12, 17, 4)):
                main = 1 if slot == 5 else 3
                materialized = torch.cat([initial[main].to(x.dtype), x[start:end].T], 1)
                self.assertTrue(torch.equal(state[slot], materialized[:, offset:offset + 9].float()))

    def test_unsupported_layouts_and_modes_fall_back(self):
        x, weight, state = _inputs(3, 65, 5, 4, 3, 11)
        meta = _metadata([1, 2], [2, 4])
        args = (x, weight, state, meta[3], meta[1], meta[2], 3)
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            self.assertFalse(_use_qwen4_varlen_prefill(mode, meta[0], *args))
        self.assertFalse(_use_qwen4_varlen_prefill(ForwardMode.EXTEND, None, *args))
        self.assertFalse(_use_qwen4_varlen_prefill(
            ForwardMode.EXTEND, meta[0], *args,
            track=(torch.tensor([1, 3], device="cuda", dtype=torch.int32), meta[3]),
        ))
        for index, invalid in (
            (0, x.T.contiguous().T),
            (1, weight.float()),
            (2, state.transpose(1, 2).contiguous().transpose(1, 2)),
            (2, state.half()),
        ):
            kernel_args = [x, weight, state, meta[3], *meta[:3], 3]
            kernel_args[index] = invalid
            self.assertFalse(can_fuse_qwen4_varlen_conv(*kernel_args))

    def test_graph_replay_reads_live_packed_metadata(self):
        x, weight, initial = _inputs(8, 65, 8, 4, 3, 12)
        state = initial.clone()
        meta = _metadata([8, 0, 0], [1, 0, 0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_qwen4_varlen_conv(x, weight, state, meta[3], *meta[:3], 3)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fused_qwen4_varlen_conv(x, weight, state, meta[3], *meta[:3], 3)
        for lengths, slots in (([3, 0, 5], [2, 0, 4]), ([0, 4, 4], [0, 3, 5])):
            live = _metadata(lengths, slots)
            for destination, source in zip(meta, live):
                destination.copy_(source)
            state.copy_(initial)
            graph.replay()
            expected, expected_state = _reference(x, weight, initial, meta[0], meta[3], 3)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.05)
            self.assertTrue(torch.equal(state, expected_state))


if __name__ == "__main__":
    unittest.main()