from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from .utils import *

F = torch.nn.functional

if TYPE_CHECKING:
    import comfy


class WindowSize(NamedTuple):
    height: int
    width: int

    @property
    def sum(self):
        return self.height * self.width

    def __neg__(self):
        return self.__class__(-self.height, -self.width)


class ShiftSize(WindowSize):
    pass


class ApplyMSWMSAAttention:
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node applies an attention patch which _may_ slightly improve quality especially when generating at high resolutions. It is a large performance increase on SD1.x, may improve performance on SDXL. This is the advanced version of the node with more parameters, use ApplyMSWMSAAttentionSimple if this seems too complex. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_blocks": (
                    "STRING",
                    {
                        "default": "1,2",
                        "tooltip": "Comma-separated list of input blocks to patch. Default is for SD1.x, you can try 4,5 for SDXL",
                    },
                ),
                "middle_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated list of middle blocks to patch. Generally not recommended.",
                    },
                ),
                "output_blocks": (
                    "STRING",
                    {
                        "default": "9,10,11",
                        "tooltip": "Comma-separated list of output blocks to patch. Default is for SD1.x, you can try 3,4,5 for SDXL",
                    },
                ),
                "time_mode": (
                    (
                        "percent",
                        "timestep",
                        "sigma",
                    ),
                    {
                        "tooltip": "Time mode controls how to interpret the values in start_time and end_time.",
                    },
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time the MSW-MSA attention effect starts applying - value is inclusive.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time the MSW-MSA attention effect ends - value is inclusive.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch with the MSW-MSA attention effect.",
                    },
                ),
            },
        }

    # reference: https://github.com/microsoft/Swin-Transformer
    # Window functions adapted from https://github.com/megvii-research/HiDiffusion
    @staticmethod
    def window_partition(
        x: torch.Tensor,
        window_size: WindowSize,
        shift_size: ShiftSize,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, _features, channels = x.shape
        wheight, wwidth = window_size
        x = x.view(batch, height, width, channels)
        if height % 2 != 0 or width % 2 != 0:
            x = F.interpolate(x.permute(0, 3, 1, 2).contiguous(), size=(wheight * 2, wwidth * 2), mode="nearest-exact").permute(0, 2, 3, 1).contiguous()
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=-shift_size, dims=(1, 2))
        x = x.view(
            batch,
            2,
            wheight,
            2,
            wwidth,
            channels,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size.height, window_size.width, channels)
        )
        return windows.view(-1, window_size.sum, channels)

    @staticmethod
    def window_reverse(
        windows: torch.Tensor,
        window_size: WindowSize,
        shift_size: WindowSize,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, _features, channels = windows.shape
        wheight, wwidth = window_size
        windows = windows.view(-1, wheight, wwidth, channels)
        batch = int(windows.shape[0] / 4)
        x = windows.view(batch, 2, 2, wheight, wwidth, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, wheight * 2, wwidth * 2, -1)
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))
        if height % 2 != 0 or width % 2 != 0:
            x = F.interpolate(x.permute(0, 3, 1, 2).contiguous(), size=(height, width), mode="nearest-exact").permute(0, 2, 3, 1).contiguous()
        return x.view(batch, height * width, channels)

    @staticmethod
    def get_window_args(
        n: torch.Tensor,
        orig_shape: tuple,
        shift: int,
    ) -> tuple[WindowSize, ShiftSize, int, int]:
        _batch, features, _channels = n.shape
        orig_height, orig_width = orig_shape[-2:]

        width, height = rescale_size(orig_width, orig_height, features)
        wheight, wwidth = math.ceil(height / 2), math.ceil(width / 2)

        if shift == 0:
            shift_size = ShiftSize(0, 0)
        elif shift == 1:
            shift_size = ShiftSize(wheight // 4, wwidth // 4)
        elif shift == 2:
            shift_size = ShiftSize(wheight // 4 * 2, wwidth // 4 * 2)
        else:
            shift_size = ShiftSize(wheight // 4 * 3, wwidth // 4 * 3)
        return (WindowSize(wheight, wwidth), shift_size, height, width)

    @classmethod
    def patch(
        cls,
        *,
        model: comfy.model_patcher.ModelPatcher,
        input_blocks: str,
        middle_blocks: str,
        output_blocks: str,
        time_mode: str,
        start_time: float,
        end_time: float,
    ) -> tuple[comfy.model_patcher.ModelPatcher]:
        use_blocks = parse_blocks("input", input_blocks)
        use_blocks |= parse_blocks("middle", middle_blocks)
        use_blocks |= parse_blocks("output", output_blocks)

        model = model.clone()
        if not use_blocks:
            return (model,)

        window_args = last_block = last_shift = None

        ms = model.get_model_object("model_sampling")

        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)

        def attn1_patch(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            extra_options: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal window_args, last_shift, last_block
            window_args = None
            last_block = extra_options.get("block")
            if last_block not in use_blocks or not check_time(
                extra_options,
                start_sigma,
                end_sigma,
            ):
                return q, k, v
            orig_shape = extra_options["original_shape"]
            # MSW-MSA
            shift = int(torch.rand(1, device="cpu").item() * 4)
            if shift == last_shift:
                shift = (shift + 1) % 4
            last_shift = shift
            window_args = tuple(
                cls.get_window_args(x, orig_shape, shift) if x is not None else None
                for x in (q, k, v)
            )
            if q is not None and q is k and q is v:
                return (
                    cls.window_partition(
                        q,
                        *window_args[0],
                    ),
                ) * 3
            return tuple(
                cls.window_partition(x, *window_args[idx])
                if x is not None
                else None
                for idx, x in enumerate((q, k, v))
            )

        def attn1_output_patch(n: torch.Tensor, extra_options: dict) -> torch.Tensor:
            nonlocal window_args
            if window_args is None or last_block != extra_options.get("block"):
                window_args = None
                return n
            args, window_args = window_args[0], None
            return cls.window_reverse(n, *args)

        model.set_model_attn1_patch(attn1_patch)
        model.set_model_attn1_output_patch(attn1_output_patch)
        return (model,)


class ApplyMSWMSAAttentionSimple:
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION = "go"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node applies an attention patch which _may_ slightly improve quality especially when generating at high resolutions. It is a large performance increase on SD1.x, may improve performance on SDXL. This is the simplified version of the node with less parameters. Use ApplyMSWMSAAttention if you require more control. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_type": (
                    ("SD15", "SDXL"),
                    {
                        "tooltip": "Model type being patched. Choose SD15 for SD 1.4, SD 2.x.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch with the MSW-MSA attention effect.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        model_type: str,
        model: comfy.model_patcher.ModelPatcher,
    ) -> tuple[comfy.model_patcher.ModelPatcher]:
        time_range = (0.2, 1.0)
        if model_type == "SD15":
            blocks = ("1,2", "", "11,10,9")
        elif model_type == "SDXL":
            blocks = ("4,5", "", "3,4,5")
        else:
            raise ValueError("Unknown model type")
        prettyblocks = " / ".join(b or "none" for b in blocks)
        logging.info(
            f"** ApplyMSWMSAAttentionSimple: Using preset {model_type}: in/mid/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}",
        )
        return ApplyMSWMSAAttention.patch(
            model=model,
            input_blocks=blocks[0],
            middle_blocks=blocks[1],
            output_blocks=blocks[2],
            time_mode="percent",
            start_time=time_range[0],
            end_time=time_range[1],
        )


__all__ = ("ApplyMSWMSAAttention", "ApplyMSWMSAAttentionSimple")
