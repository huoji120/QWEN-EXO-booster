"""Unit tests for ``BaseMultimodalProcessor._load_single_item`` image decoding.

Regression test for the change that forces the (otherwise lazy) PIL decode inside
``_load_single_item`` — which runs in the ``io_executor`` worker thread — instead of
letting it fire lazily on the main event-loop thread later (inside
``pil_to_tensor``/``tobytes`` during processing). The behavior of the returned image
(mode, pixels) must be unchanged; only *when/where* the decode happens differs.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


def _png_bytes(mode: str = "RGB", size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_decoded(img: Image.Image) -> bool:
    """A lazily-opened PIL image has no decoded core yet; ``load()`` populates it.
    PIL's ``.im`` property requires a completed load and raises otherwise."""
    try:
        return img.im is not None
    except Exception:
        return False


class TestLoadSingleItemImageDecode(CustomTestCase):
    def test_plain_open_is_lazy(self):
        # Documents why the fix matters: a bare Image.open is not decoded yet, so
        # without the fix the decode would land on the caller (main) thread.
        lazy = Image.open(io.BytesIO(_png_bytes()))
        self.assertFalse(_is_decoded(lazy))

    def test_load_single_item_forces_decode(self):
        img = BaseMultimodalProcessor._load_single_item(
            _png_bytes("RGB"), Modality.IMAGE, gpu_image_decode=False
        )
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        # The fix: decode is forced inside _load_single_item, not lazily later.
        self.assertTrue(_is_decoded(img))

    def test_rgba_converted_to_rgb_and_decoded(self):
        img = BaseMultimodalProcessor._load_single_item(
            _png_bytes("RGBA"), Modality.IMAGE, gpu_image_decode=False
        )
        # Existing alpha-discard behavior preserved.
        self.assertEqual(img.mode, "RGB")
        self.assertTrue(_is_decoded(img))

    def test_pixels_match_reference(self):
        # Output must be bit-identical to the pre-fix path (open -> [convert]).
        data = _png_bytes("RGB")
        img = BaseMultimodalProcessor._load_single_item(
            data, Modality.IMAGE, gpu_image_decode=False
        )
        ref = Image.open(io.BytesIO(data)).convert("RGB")
        np.testing.assert_array_equal(np.asarray(img), np.asarray(ref))


class _ConfiguredProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError("These tests exercise decoding and preprocessing directly")


class TestCPUImageProcessing(CustomTestCase):
    def setUp(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
            pad_token="[UNK]",
            additional_special_tokens=[
                "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
            ],
        )
        self.hf = Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                min_pixels=56 * 56, max_pixels=112 * 112
            ),
            tokenizer=tokenizer,
            video_processor=Qwen2VLVideoProcessor(),
        )
        self.processor = _ConfiguredProcessor(
            SimpleNamespace(),
            SimpleNamespace(
                keep_mm_feature_on_device=False,
                mm_feature_transport="cpu",
                disable_fast_image_processor=False,
                skip_tokenizer_init=False,
                mm_process_config={"image": {"device": "cpu"}},
            ),
            self.hf,
            transport_mode="cpu",
        )
        self.addCleanup(self.processor.io_executor.shutdown)
        self.addCleanup(self.processor.cpu_executor.shutdown)

    def test_cpu_device_preserves_repeated_image_pixels_grids_and_tokens(self):
        from sglang.srt.multimodal.processors import base_processor as module

        image = Image.open(io.BytesIO(_png_bytes(size=(56, 56))))
        images = [image, image, image]
        text = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        with (
            patch.multiple(module, _is_cpu=False, _is_xpu=False, _is_npu=False),
            patch.object(
                module,
                "get_server_args",
                return_value=SimpleNamespace(rl_on_policy_target=None, base_gpu_id=0),
            ),
        ):
            actual = self.processor.process_mm_data(text, images=images)
        expected = self.hf(
            text=[text],
            images=images,
            images_kwargs={"device": "cpu"},
            padding=True,
            return_tensors="pt",
        )
        for key in ("pixel_values", "image_grid_thw", "input_ids"):
            np.testing.assert_array_equal(actual[key].numpy(), expected[key].numpy())
        self.assertEqual(actual["image_grid_thw"].shape[0], len(images))
        self.assertEqual(actual["pixel_values"].device.type, "cpu")

    def test_cpu_device_decodes_jpeg_without_nvjpeg(self):
        from sglang.srt.utils import common

        buf = io.BytesIO()
        Image.open(io.BytesIO(_png_bytes(size=(56, 56)))).save(buf, format="JPEG")
        data = buf.getvalue()
        with patch.object(common, "is_cuda", return_value=True), patch.object(
            common,
            "decode_jpeg",
            side_effect=AssertionError("CPU images must not allocate nvJPEG GPU memory"),
        ) as gpu_decoder:
            jobs = self.processor._submit_mm_data_loading_tasks_simple(
                [data], Modality.IMAGE, None, True
            )
            actual = jobs[0][2].result(timeout=5)
        gpu_decoder.assert_not_called()
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        )


if __name__ == "__main__":
    unittest.main()
