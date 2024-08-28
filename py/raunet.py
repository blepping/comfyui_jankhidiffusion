from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import torch
from comfy.ldm.modules.diffusionmodules import openaimodel

from .utils import (
    UPSCALE_METHODS,
    check_time,
    convert_time,
    get_sigma,
    parse_blocks,
    scale_samples,
)

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

F = torch.nn.functional


class HDConfig:
    def __init__(
        self,
        start_sigma: float,
        end_sigma: float,
        use_blocks: dict,
        upscale_mode: str,
        two_stage_upscale_mode: str,
    ):
        self.curr_sigma: None | float = None
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.use_blocks = use_blocks
        self.upscale_mode = upscale_mode
        self.two_stage_upscale_mode = two_stage_upscale_mode

    def check(self, topts: dict) -> bool:
        if not isinstance(topts, dict) or topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)

    def __str__(self):
        return f"<HDConfig: curr_sigma={self.curr_sigma}, start_sigma={self.start_sigma}, end_sigma={self.end_sigma}, use_blocks={self.use_blocks!r}, upscale_mode={self.upscale_mode}, two_stage_upscale_mode={self.two_stage_upscale_mode}>"


GLOBAL_STATE: HDState


class HDState:
    def __init__(self):
        self.no_controlnet_workaround = (
            "JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND" in os.environ
        )
        self.controlnet_scale_args = {"mode": "bilinear", "align_corners": False}
        self.patched_freeu_advanced = False
        self.orig_apply_control = openaimodel.apply_control
        self.orig_fua_apply_control = None

    def hd_apply_control(
        self,
        h: torch.Tensor,
        control: None | dict,
        name: str,
    ) -> torch.Tensor:
        ctrls = control.get(name) if control is not None else None
        if ctrls is None or len(ctrls) == 0:
            return h
        ctrl = ctrls.pop()
        if ctrl is None:
            return h
        if ctrl.shape[-2:] != h.shape[-2:]:
            logging.info(
                f"* jankhidiffusion: Scaling controlnet conditioning: {ctrl.shape[-2:]} -> {h.shape[-2:]}",
            )
            ctrl = F.interpolate(ctrl, size=h.shape[-2:], **self.controlnet_scale_args)
        h += ctrl
        return h

    def try_patch_apply_control(self) -> None:
        if (
            self.no_controlnet_workaround
            or openaimodel.apply_control == self.hd_apply_control
        ):
            return
        self.orig_apply_control = openaimodel.apply_control
        openaimodel.apply_control = self.hd_apply_control
        logging.info("** jankhidiffusion: Patched openaimodel.apply_control")

    # Try to be compatible with FreeU Advanced.
    def try_patch_freeu_advanced(self) -> None:
        if self.patched_freeu_advanced or self.no_controlnet_workaround:
            return

        # We only try one time.
        self.patched_freeu_advanced = True
        fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
        if not fua_nodes:
            return

        self.orig_fua_apply_control = fua_nodes.apply_control
        fua_nodes.apply_control = self.hd_apply_control
        logging.info("** jankhidiffusion: Patched FreeU_Advanced")

    def apply_patches(self) -> None:
        self.try_patch_apply_control()
        self.try_patch_freeu_advanced()

    def revert_patches(self) -> None:
        if openaimodel.apply_control == self.hd_apply_control:
            openaimodel.apply_control = self.orig_apply_control
            logging.info("** jankhidiffusion: Reverted openaimodel.apply_control patch")
        if not self.patched_freeu_advanced:
            return
        fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
        if not fua_nodes:
            logging.warning(
                "** jankhidiffusion: Unexpectedly could not revert FreeU_Advanced patches",
            )
            return
        fua_nodes.apply_control = self.orig_fua_apply_control
        self.patched_freeu_advanced = False
        logging.info("** jankhidiffusion: Reverted FreeU_Advanced patch")


GLOBAL_STATE = HDState()


class HDForward:
    FORWARD_DOWNSAMPLE_COPY_OP_KEYS = (
        "comfy_cast_weights",
        "weight_function",
        "bias_function",
        "weight",
        "bias",
    )

    def __init__(
        self,
        orig_block: object,
        hdconfig: HDConfig,
        block_index: int,
        is_up: bool,
    ):
        self.orig_block = orig_block
        orig_forward = orig_block.forward
        # This is weird but apparently when we patch the model, the previous object patches
        # may still exist, so we have to make sure we get the _real_ original forward function.
        while isinstance(orig_forward, HDForward):
            orig_forward = orig_forward.orig_forward
        self.orig_forward = orig_forward
        self.hdconfig = hdconfig
        self.block_index = block_index
        self.forward = self.forward_upsample if is_up else self.forward_downsample

    def __call__(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward_upsample(
        self,
        x: torch.Tensor,
        output_shape: None | tuple = None,
    ) -> torch.Tensor:
        hdconfig = self.hdconfig
        orig_block = self.orig_block
        block_index = self.block_index
        if (
            orig_block.dims == 3
            or not orig_block.use_conv
            or not hdconfig.check({
                "sigmas": hdconfig.curr_sigma,
                "block": ("output", block_index),
            })
        ):
            return self.orig_forward(x, output_shape=output_shape)

        shape = (
            output_shape[2:4]
            if output_shape is not None
            else (x.shape[2] * 4, x.shape[3] * 4)
        )
        if hdconfig.two_stage_upscale_mode != "disabled":
            x = scale_samples(
                x,
                shape[1] // 2,
                shape[0] // 2,
                mode=hdconfig.two_stage_upscale_mode,
                sigma=hdconfig.curr_sigma,
            )
        x = scale_samples(
            x,
            shape[1],
            shape[0],
            mode=hdconfig.upscale_mode,
            sigma=hdconfig.curr_sigma,
        )
        return orig_block.conv(x)

    def forward_downsample(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hdconfig = self.hdconfig
        orig_block = self.orig_block
        block_index = self.block_index
        if (
            orig_block.dims == 3
            or not orig_block.use_conv
            or not hdconfig.check({
                "sigmas": hdconfig.curr_sigma,
                "block": ("input", block_index),
            })
        ):
            return self.orig_forward(x)

        tempop = openaimodel.ops.conv_nd(
            orig_block.dims,
            orig_block.channels,
            orig_block.out_channels,
            3,  # kernel size
            stride=(4, 4),
            padding=(2, 2),
            dilation=(2, 2),
            dtype=x.dtype,
            device=x.device,
        )

        if (
            orig_block.op.__class__.__base__ is not None
            and orig_block.op.__class__.__base__.__name__ == "GGMLLayer"
        ):
            # Workaround for GGML quantized Downsample blocks.
            if not hasattr(orig_block.op, "get_weights"):
                errstr = f"Cannot handle downsample block {block_index} which appears to be GGUF quantized but has no get_weights method!"
                raise RuntimeError(errstr)
            tempop.comfy_cast_weights = True
            tempop.weight, tempop.bias = (
                torch.nn.Parameter(p).to(device=x.device)
                for p in orig_block.op.get_weights(x.dtype)
            )
            return tempop(x)

        for k in self.FORWARD_DOWNSAMPLE_COPY_OP_KEYS:
            setattr(tempop, k, getattr(orig_block.op, k))
        return tempop(x)


class ApplyRAUNet:
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the RAUNet effect.",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node is used to enable generation at higher resolutions than a model was trained for with less artifacts or other negative effects. This is the advanced version with more tuneable parameters, use ApplyRAUNetSimple if this seems too complex. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to be patched with the RAUNet effect.",
                    },
                ),
                "input_blocks": (
                    "STRING",
                    {
                        "default": "3",
                        "tooltip": "Comma-separated list of input Downsample blocks. Default is for SD 1.5. The corresponding valid block from output_blocks must be set along with input.\nValid blocks for SD1.5: 3, 6, 9\nValid blocks for SDXL: 3, 6",
                    },
                ),
                "output_blocks": (
                    "STRING",
                    {
                        "default": "8",
                        "tooltip": "Comma-separated list of output Upsample blocks. Default is for SD 1.5. The corresponding valid block from input_blocks must be set along with output.\nValid blocks for SD1.5: 8, 5, 2\nValid blocks for SDXL: 5, 2",
                    },
                ),
                "time_mode": (
                    ("percent", "timestep", "sigma"),
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
                        "tooltip": "Time normal RAUNet effects start applying - value is inclusive.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time normal RAUNet effects end - value is inclusive.",
                    },
                ),
                "upscale_mode": (
                    UPSCALE_METHODS,
                    {
                        "tooltip": "Method used when upscaling latents in output Upscale blocks.",
                    },
                ),
                "ca_start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time normal cross-attention effects start applying - value is inclusive..",
                    },
                ),
                "ca_end_time": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time normal cross-attention effects end - value is inclusive.",
                    },
                ),
                "ca_input_blocks": (
                    "STRING",
                    {
                        "default": "4",
                        "tooltip": "Comma separated list of input cross-attention blocks. Default is for SD1.x, for SDXL you can try using 2 (or just disable it).",
                    },
                ),
                "ca_output_blocks": (
                    "STRING",
                    {
                        "default": "8",
                        "tooltip": "Comma-separated list of output cross-attention blocks. Default is for SD1.x, for SDXL you can try using 7 (or just disable it).",
                    },
                ),
                "ca_upscale_mode": (
                    UPSCALE_METHODS,
                    {
                        "tooltip": "Mode used when upscaling latents in output cross-attention blocks.",
                    },
                ),
                "ca_downscale_mode": (
                    ("avg_pool2d", *UPSCALE_METHODS),
                    {
                        "default": "avg_pool2d",
                        "tooltip": "Mode used when downscaling latents in output cross-attention blocks (use avg_pool2d for normal Hidiffusion behavior).",
                    },
                ),
                "ca_downscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "step": 0.1,
                        "round": False,
                        "tooltip": "Factor to downscale with in cross-attention, 2.0 means downscale to half size. Must be an integer when using ca_downscale_mode avg_pool2d.",
                    },
                ),
                "two_stage_upscale_mode": (
                    ("disabled", *UPSCALE_METHODS),
                    {
                        "default": "disabled",
                        "tooltip": "When upscaling in output Upscale blocks (non-NA), do half the upscale with this mode and half with the normal upscale mode. May produce a different effect, isn't necessarily better.",
                    },
                ),
            },
        }

    @classmethod
    def patch(
        cls,
        *,
        model: ModelPatcher,
        input_blocks: str,
        output_blocks: str,
        time_mode: str,
        start_time: float,
        end_time: float,
        upscale_mode: str,
        ca_start_time: float,
        ca_end_time: float,
        ca_input_blocks: str,
        ca_output_blocks: str,
        ca_upscale_mode: str,
        ca_downscale_mode: str = "avg_pool2d",
        ca_downscale_factor: float = 2.0,
        two_stage_upscale_mode: str = "disabled",
    ) -> tuple[ModelPatcher]:
        if ca_downscale_mode == "avg_pool2d" and not ca_downscale_factor.is_integer():
            raise ValueError(
                "avg_pool2d downscale mode can only be used with integer downscale factors",
            )
        use_blocks = parse_blocks("output", output_blocks)
        use_blocks |= parse_blocks("input", input_blocks)

        ca_use_blocks = parse_blocks("output", ca_output_blocks)
        have_ca_output_blocks = len(ca_use_blocks) > 0
        ca_use_blocks |= parse_blocks("input", ca_input_blocks)

        model = model.clone()
        ms = model.get_model_object("model_sampling")

        ca_start_sigma, ca_end_sigma = convert_time(
            ms,
            time_mode,
            ca_start_time,
            ca_end_time,
        )

        hdconfig = HDConfig(
            *convert_time(
                ms,
                time_mode,
                start_time,
                end_time,
            ),
            use_blocks,
            upscale_mode,
            two_stage_upscale_mode,
        )

        def input_block_patch(h: torch.Tensor, extra_options: dict) -> torch.Tensor:
            block_type, block_index = extra_options.get("block", ("unknown", -1))
            if block_index == 0:
                hdconfig.curr_sigma = get_sigma(extra_options)
            if (block_type, block_index) not in ca_use_blocks or not check_time(
                hdconfig.curr_sigma,
                ca_start_sigma,
                ca_end_sigma,
            ):
                return h
            if ca_downscale_mode == "avg_pool2d":
                return F.avg_pool2d(
                    h,
                    kernel_size=(int(ca_downscale_factor), int(ca_downscale_factor)),
                )
            return scale_samples(
                h,
                max(1, int(h.shape[-1] // ca_downscale_factor)),
                max(1, int(h.shape[-2] // ca_downscale_factor)),
                mode=ca_downscale_mode,
                sigma=hdconfig.curr_sigma,
            )

        def output_block_patch(
            h: torch.Tensor,
            hsp: torch.Tensor,
            extra_options: dict,
        ) -> torch.Tensor:
            if extra_options.get("block") not in ca_use_blocks or not check_time(
                hdconfig.curr_sigma,
                ca_start_sigma,
                ca_end_sigma,
            ):
                return h, hsp
            sigma = hdconfig.curr_sigma
            block = extra_options.get("block", ("", 0))[1]
            if sigma is not None and (block < 3 or block > 6):
                sigma /= 16
            return scale_samples(
                h,
                hsp.shape[-1],
                hsp.shape[-2],
                mode=ca_upscale_mode,
                sigma=sigma,
            ), hsp

        model.set_model_input_block_patch(input_block_patch)
        if have_ca_output_blocks:
            model.set_model_output_block_patch(output_block_patch)

        for block_type, block_index in use_blocks:
            main_block = model.get_model_object(
                f"diffusion_model.{block_type}_blocks.{block_index}",
            )
            expected_class = (
                openaimodel.Downsample
                if block_type == "input"
                else openaimodel.Upsample
            )
            block_name = f"diffusion_model.{block_type}_blocks.{block_index}.{len(main_block) - 1}"
            block = model.get_model_object(block_name)
            if not isinstance(block, expected_class):
                block_type_name = getattr(type(block), "__name__", "unknown")
                error_message = (
                    f"User error: {block_type} {block_index} requires targeting an {expected_class.__name__} block but got block of type {block_type_name} instead.",
                )
                raise ValueError(error_message)  # noqa: TRY004
            model.add_object_patch(
                f"{block_name}.forward",
                HDForward(block, hdconfig, block_index, block_type != "input"),
            )

        GLOBAL_STATE.apply_patches()

        return (model,)


class ApplyRAUNetSimple:
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the RAUNet effect.",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node is used to enable generation at higher resolutions than a model was trained for with less artifacts or other negative effects. This is the simplified version with less parameters, use ApplyRAUNet if you require more control. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to be patched with the RAUNet effect.",
                    },
                ),
                "model_type": (
                    ("SD15", "SDXL"),
                    {
                        "tooltip": "Model type being patched. Choose SD15 for SD 1.4 or SD 2.x.",
                    },
                ),
                "res_mode": (
                    (
                        "high (1536-2048)",
                        "low (1024 or lower)",
                        "ultra (over 2048)",
                    ),
                    {
                        "tooltip": "Resolution mode hint, does not have to correspond to the actual size.",
                    },
                ),
                "upscale_mode": (
                    (
                        "default",
                        *UPSCALE_METHODS,
                    ),
                    {
                        "tooltip": "Method used when upscaling latents in output Upsample blocks.",
                    },
                ),
                "ca_upscale_mode": (
                    (
                        "default",
                        *UPSCALE_METHODS,
                    ),
                    {
                        "tooltip": "Method used when upscaling latents in cross attention blocks.",
                    },
                ),
            },
        }

    @classmethod
    def patch(
        cls,
        *,
        model: ModelPatcher,
        model_type: str,
        res_mode: str,
        upscale_mode: str,
        ca_upscale_mode: str,
    ) -> tuple[ModelPatcher]:
        if upscale_mode == "default":
            upscale_mode = "bicubic"
        if ca_upscale_mode == "default":
            ca_upscale_mode = "bicubic"
        res = res_mode.split(" ", 1)[0]
        if model_type == "SD15":
            blocks = ("3", "8")
            ca_blocks = ("1", "11")
            time_range = (0.0, 0.6)
            if res == "low":
                time_range = (0.0, 0.4)
                ca_time_range = (1.0, 0.0)
                ca_blocks = ("", "")
            elif res == "high":
                time_range = (0.0, 0.5)
                ca_time_range = (0.0, 0.35)
            elif res == "ultra":
                time_range = (0.0, 0.6)
                ca_time_range = (0.0, 0.45)
            else:
                raise ValueError("Unknown res_mode")
        elif model_type == "SDXL":
            blocks = ("3", "5")
            ca_blocks = ("4", "5")
            if res == "low":
                time_range = (1.0, 0.0)
                ca_time_range = (1.0, 0.0)
                ca_blocks = ("", "")
            elif res == "high":
                time_range = (0.0, 0.5)
                ca_time_range = (1.0, 0.0)
            elif res == "ultra":
                time_range = (0.0, 0.6)
                ca_time_range = (0.0, 0.45)
            else:
                raise ValueError("Unknown res_mode")
        else:
            raise ValueError("Unknown model type")

        prettyblocks = " / ".join(b or "none" for b in blocks)
        prettycablocks = " / ".join(b or "none" for b in ca_blocks)
        logging.info(
            f"** ApplyRAUNetSimple: Using preset {model_type} {res}: upscale {upscale_mode}, in/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}  |  CA upscale {ca_upscale_mode},  CA in/out blocks [{prettycablocks}], CA start/end percent {ca_time_range[0]:.2}/{ca_time_range[1]:.2}",
        )
        return ApplyRAUNet.patch(
            model=model,
            input_blocks=blocks[0],
            output_blocks=blocks[1],
            time_mode="percent",
            start_time=time_range[0],
            end_time=time_range[1],
            upscale_mode=upscale_mode,
            ca_start_time=ca_time_range[0],
            ca_end_time=ca_time_range[1],
            ca_input_blocks=ca_blocks[0],
            ca_output_blocks=ca_blocks[1],
            ca_upscale_mode=ca_upscale_mode,
        )


__all__ = ("ApplyRAUNet", "ApplyRAUNetSimple")
