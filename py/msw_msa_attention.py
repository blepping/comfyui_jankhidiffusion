from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import torch

from .utils import *

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
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_blocks": ("STRING", {"default": "1,2"}),
                "middle_blocks": ("STRING", {"default": ""}),
                "output_blocks": ("STRING", {"default": "9,10,11"}),
                "time_mode": (
                    (
                        "percent",
                        "timestep",
                        "sigma",
                    ),
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
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
                    },
                ),
                "model": ("MODEL",),
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
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=-shift_size, dims=(1, 2))
        x = x.view(
            batch,
            height // wheight,
            wheight,
            width // wwidth,
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
        batch = int(
            windows.shape[0] / (height * width / wheight / wwidth),
        )
        x = windows.view(batch, height // wheight, width // wwidth, wheight, wwidth, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))
        return x.view(batch, height * width, channels)

    @staticmethod
    def get_window_args(
        n: torch.Tensor,
        orig_shape: tuple,
        shift: int,
    ) -> tuple[WindowSize, ShiftSize, int, int]:
        _batch, features, _channels = n.shape
        orig_height, orig_width = orig_shape[-2:]

        downsample_ratio = int(
            ((orig_height * orig_width) // features) ** 0.5,
        )
        height, width = (
            orig_height // downsample_ratio,
            orig_width // downsample_ratio,
        )
        wheight, wwidth = height // 2, width // 2

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
            try:
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
            except RuntimeError as exc:
                logging.warning(
                    f"** jankhidiffusion: MSW-MSA attention not applied: Incompatible model patches or bad resolution. Try using resolutions that are multiples of 32 or 64. Original exception: {exc}",
                )
                window_args = None
                return q, k, v

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
    FUNCTION = "go"
    CATEGORY = "model_patches/unet"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_type": (("SD15", "SDXL"),),
                "model": ("MODEL",),
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
            blocks = ("4,5", "", "5,4")
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
