from __future__ import annotations

import argparse
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from diffusers import DiffusionPipeline

from lingbot_video import runner
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline


class _Visual(torch.nn.Module):
    def forward(self, value):
        return value


class _TextEncoderCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()

    def forward(self, **kwargs):
        return kwargs


class _TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TextEncoderCore()


class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class _Processor:
    pass


def _minimal_pipeline() -> LingBotVideoPipeline:
    return LingBotVideoPipeline(
        transformer=torch.nn.Linear(1, 1),
        vae=_Vae(),
        text_encoder=_TextEncoder(),
        processor=_Processor(),
        scheduler=None,
    )


class CpuOffloadPipelineTests(unittest.TestCase):
    def test_sequential_offload_preloads_qwen_visual_module(self):
        pipe = _minimal_pipeline()

        def enable_super(instance, gpu_id=None, device=None):
            instance._offload_device = torch.device(device or "cuda:0")

        with (
            patch.object(
                DiffusionPipeline,
                "enable_sequential_cpu_offload",
                autospec=True,
                side_effect=enable_super,
            ),
            patch("accelerate.hooks.remove_hook_from_module") as remove_hook,
            patch("accelerate.cpu_offload") as cpu_offload,
        ):
            pipe.enable_sequential_cpu_offload(device="cuda:0")

        remove_hook.assert_called_once_with(pipe.text_encoder, recurse=True)
        cpu_offload.assert_called_once_with(
            pipe.text_encoder,
            execution_device=torch.device("cuda:0"),
            offload_buffers=False,
            preload_module_classes=["_Visual"],
        )

    def test_vae_context_uses_hook_device_and_resets_model_chain(self):
        pipe = _minimal_pipeline()
        pipe.vae._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:3"))
        pipe._all_hooks = [object()]
        pipe.maybe_free_model_hooks = Mock()

        with pipe._vae_encode_context() as device:
            self.assertEqual(device, torch.device("cuda:3"))

        self.assertEqual(pipe.maybe_free_model_hooks.call_count, 2)


class CpuOffloadRunnerTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides) -> argparse.Namespace:
        values = {
            "cpu_offload": "model",
            "engine": "diffusers",
            "enable_fsdp_inference": False,
            "enable_vlm_fsdp_inference": False,
            "cfg_parallel_degree": 1,
            "context_parallel_degree": 1,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_validation_accepts_single_gpu_diffusers(self):
        with (
            patch.object(runner, "_distributed_env", return_value=(0, 0, 1)),
            patch.object(runner.torch.cuda, "is_available", return_value=True),
        ):
            runner._validate_cpu_offload_args(self._args())

    def test_validation_leaves_default_runtime_unchanged(self):
        with (
            patch.object(runner, "_distributed_env") as distributed_env,
            patch.object(runner.torch.cuda, "is_available") as cuda_available,
        ):
            runner._validate_cpu_offload_args(self._args(cpu_offload="none"))

        distributed_env.assert_not_called()
        cuda_available.assert_not_called()

    def test_validation_rejects_each_incompatible_runtime(self):
        cases = (
            ({"engine": "sglang-native"}, (0, 0, 1)),
            ({"enable_fsdp_inference": True}, (0, 0, 1)),
            ({"enable_vlm_fsdp_inference": True}, (0, 0, 1)),
            ({"cfg_parallel_degree": 2}, (0, 0, 2)),
            ({"context_parallel_degree": 2}, (0, 0, 2)),
            ({}, (0, 0, 2)),
        )
        for overrides, distributed_env in cases:
            with (
                self.subTest(overrides=overrides, distributed_env=distributed_env),
                patch.object(runner, "_distributed_env", return_value=distributed_env),
                patch.object(runner.torch.cuda, "is_available", return_value=True),
                self.assertRaises(ValueError),
            ):
                runner._validate_cpu_offload_args(self._args(**overrides))

    def test_validation_rejects_missing_cuda(self):
        with (
            patch.object(runner, "_distributed_env", return_value=(0, 0, 1)),
            patch.object(runner.torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "requires CUDA"),
        ):
            runner._validate_cpu_offload_args(self._args())

    def test_loader_enables_requested_offload_without_moving_whole_pipeline(self):
        for mode, expected_method in (
            ("model", "enable_model_cpu_offload"),
            ("sequential", "enable_sequential_cpu_offload"),
        ):
            with self.subTest(mode=mode):
                pipe = SimpleNamespace(
                    enable_model_cpu_offload=Mock(),
                    enable_sequential_cpu_offload=Mock(),
                    to=Mock(),
                )
                pipeline_class = Mock()
                pipeline_class.from_pretrained.return_value = pipe
                with (
                    patch.object(runner, "_pipeline_class_for_mode", return_value=pipeline_class),
                    patch.object(runner, "_load_transformer_component", return_value=object()),
                    patch.object(runner, "_patch_qwen3vl_from_pretrained", return_value=nullcontext()),
                    patch.object(runner, "_default_device", return_value=torch.device("cuda:0")),
                ):
                    loaded = runner._load_diffusers_pipe(
                        Path("/model"),
                        {"default": torch.bfloat16},
                        mode="t2v",
                        transformer_subfolder="transformer",
                        cpu_offload=mode,
                    )

                self.assertIs(loaded, pipe)
                getattr(pipe, expected_method).assert_called_once_with(
                    device=torch.device("cuda:0")
                )
                pipe.to.assert_not_called()

    def test_loader_keeps_default_full_device_placement(self):
        moved_pipe = object()
        pipe = SimpleNamespace(to=Mock(return_value=moved_pipe))
        pipeline_class = Mock()
        pipeline_class.from_pretrained.return_value = pipe
        with (
            patch.object(runner, "_pipeline_class_for_mode", return_value=pipeline_class),
            patch.object(runner, "_load_transformer_component", return_value=object()),
            patch.object(runner, "_patch_qwen3vl_from_pretrained", return_value=nullcontext()),
            patch.object(runner, "_default_device", return_value=torch.device("cuda:0")),
        ):
            loaded = runner._load_diffusers_pipe(
                Path("/model"),
                {"default": torch.bfloat16},
                mode="t2v",
                transformer_subfolder="transformer",
            )

        self.assertIs(loaded, moved_pipe)
        pipe.to.assert_called_once_with(torch.device("cuda:0"))

    def test_loader_rejects_shared_or_deferred_modules_before_loading_weights(self):
        for kwargs in (
            {"deferred_components": frozenset({"transformer"})},
            {"shared_components": {"vae": object()}},
        ):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(runner, "_load_transformer_component") as load_transformer,
                self.assertRaises(ValueError),
            ):
                runner._load_diffusers_pipe(
                    Path("/model"),
                    {"default": torch.bfloat16},
                    mode="t2v",
                    transformer_subfolder="transformer",
                    cpu_offload="model",
                    **kwargs,
                )
            load_transformer.assert_not_called()

    def test_refiner_does_not_share_hooked_base_components(self):
        args = self._args(
            run_refiner=True,
            refiner_model_dir="/model",
            refiner_transformer_subfolder="refiner",
            model_dir="/model",
            refiner_vae_dtype="fp32",
            refiner_vae_tiling=True,
            refiner_vae_tile_height=None,
            refiner_vae_tile_width=None,
            refiner_vae_tile_stride_height=None,
            refiner_vae_tile_stride_width=None,
        )
        refiner_pipe = object()
        with (
            patch.object(runner, "_refiner_model_available", return_value=True),
            patch.object(runner, "_refiner_skip_reason", return_value=None),
            patch.object(runner, "_shared_auxiliary_components") as shared_components,
            patch.object(
                runner,
                "_load_pipe",
                return_value=(refiner_pipe, "diffusers-reference"),
            ) as load_pipe,
            patch.object(runner, "_component_dtypes", return_value={"vae": "float32"}),
        ):
            state = runner._maybe_preload_refiner(
                args,
                {
                    "default": torch.bfloat16,
                    "transformer": torch.bfloat16,
                    "text_encoder": torch.bfloat16,
                    "vae": torch.float32,
                },
                torch.device("cuda:0"),
                0,
                None,
                frozenset(),
                object(),
            )

        shared_components.assert_not_called()
        self.assertIs(state["pipe"], refiner_pipe)
        self.assertEqual(load_pipe.call_args.kwargs["shared_components"], {})
        self.assertTrue(load_pipe.call_args.kwargs["configure_vae_tiling"])

    def test_release_removes_hooks_before_dropping_components(self):
        pipe = SimpleNamespace(
            transformer=object(),
            text_encoder=object(),
            vae=object(),
            remove_all_hooks=Mock(),
        )
        with patch.object(runner.torch.cuda, "is_available", return_value=False):
            runner._release_pipeline_for_memory(pipe, "base")

        pipe.remove_all_hooks.assert_called_once_with()
        self.assertIsNone(pipe.transformer)
        self.assertIsNone(pipe.text_encoder)
        self.assertIsNone(pipe.vae)


if __name__ == "__main__":
    unittest.main()
