import logging
import os
import sys
from functools import partial
from inspect import signature

import torch.nn.functional as F  # noqa: N812
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ldm.modules.diffusionmodules.openaimodel import ops
from comfy.model_patcher import ModelPatcher

from .utils import (
    UPSCALE_METHODS,
    check_time,
    convert_time,
    get_sigma,
    parse_blocks,
    scale_samples,
)

NO_CONTROLNET_WORKAROUND = os.environ.get("JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND") is not None
_PATCHED_FREEU = False
_ORIG_FORWARD_TIMESTEP_EMBED = openaimodel.forward_timestep_embed
_ORIG_APPLY_CONTROL = openaimodel.apply_control


class HDConfig:
    def __init__(self, start_sigma, end_sigma, use_blocks, upscale_mode, two_stage_upscale_mode):
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.use_blocks = use_blocks
        self.upscale_mode = upscale_mode
        self.two_stage_upscale_mode = two_stage_upscale_mode

    def check(self, topts):
        if not isinstance(topts, dict) or topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)


def hd_forward_timestep_embed(ts, x, emb, *args: list, **kwargs: dict):
    transformer_options = kwargs.get("transformer_options", None)
    output_shape = kwargs.get("output_shape", None)
    transformer_options = args[1] if transformer_options is None and len(args) > 1 else {}
    output_shape = args[2] if output_shape is None and len(args) > 2 else None
    for layer in ts:
        if isinstance(layer, openaimodel.Upsample) and "transformer_options" in signature(layer.forward).parameters:
            x = layer.forward(
                x,
                output_shape=output_shape,
                transformer_options=transformer_options,
            )
        elif isinstance(layer, openaimodel.Downsample) and "transformer_options" in signature(layer.forward).parameters:
            x = layer.forward(x, transformer_options=transformer_options)
        else:
            x = _ORIG_FORWARD_TIMESTEP_EMBED((layer,), x, emb, *args, **kwargs)
    return x


def try_patch_forward_timestep_embed():
    if openaimodel.forward_timestep_embed is not hd_forward_timestep_embed:
        openaimodel.forward_timestep_embed = hd_forward_timestep_embed
        logging.info("** jankhidiffusion: Patched openaimodel.forward_timestep_embed")


def hd_apply_control(h, control, name):
    controlnet_scale_args = {"mode": "bilinear", "align_corners": False}

    ctrls = control.get(name) if control is not None else None
    if ctrls is None or len(ctrls) == 0:
        return h
    ctrl = ctrls.pop()
    if ctrl is None:
        return h
    if ctrl.shape[-2:] != h.shape[-2:]:
        logging.info(f"* jankhidiffusion: Scaling controlnet conditioning: {ctrl.shape[-2:]} -> {h.shape[-2:]}")
        ctrl = F.interpolate(ctrl, size=h.shape[-2:], **controlnet_scale_args)
    h += ctrl
    return h


def try_patch_apply_control():
    if openaimodel.apply_control is not hd_apply_control and not NO_CONTROLNET_WORKAROUND:
        openaimodel.apply_control = hd_apply_control
        logging.info("** jankhidiffusion: Patched openaimodel.apply_control")


# Try to be compatible with FreeU Advanced.
def try_patch_freeu_advanced():
    global _PATCHED_FREEU  # noqa: PLW0603
    if _PATCHED_FREEU:
        return

    # We only try one time.
    _PATCHED_FREEU = True
    fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
    if not fua_nodes:
        return

    fua_nodes.forward_timestep_embed = hd_forward_timestep_embed
    if not NO_CONTROLNET_WORKAROUND:
        fua_nodes.apply_control = hd_apply_control
    logging.info("** jankhidiffusion: Patched FreeU_Advanced")


def forward_upsample(_self, _forward, _hdconfig: HDConfig, x, output_shape=None, transformer_options=None):
    if (
        _self.dims == 3
        or not _self.use_conv
        or not _hdconfig.check(transformer_options)
    ):
        return _forward(x, output_shape=output_shape)

    shape = (
        output_shape[2:4]
        if output_shape is not None
        else (x.shape[2] * 4, x.shape[3] * 4)
    )
    if _hdconfig.two_stage_upscale_mode != "disabled":
        x = scale_samples(
            x,
            shape[1] // 2,
            shape[0] // 2,
            mode=_hdconfig.two_stage_upscale_mode,
            sigma=get_sigma(transformer_options),
        )
    x = scale_samples(
        x,
        shape[1],
        shape[0],
        mode=_hdconfig.upscale_mode,
        sigma=get_sigma(transformer_options),
    )
    return _self.conv(x)


def forward_downsample(_self, _forward, _hdconfig: HDConfig, x, transformer_options=None):
    if (
        _self.dims == 3
        or not _self.use_conv
        or not _hdconfig.check(transformer_options)
    ):
        return _forward(x)

    copy_op_keys = (
        "comfy_cast_weights",
        "weight_function",
        "bias_function",
        "weight",
        "bias",
    )
    tempop = ops.conv_nd(
        _self.dims,
        _self.channels,
        _self.out_channels,
        3,  # kernel size
        stride=(4, 4),
        padding=(2, 2),
        dilation=(2, 2),
        dtype=x.dtype,
        device=x.device,
    )
    for k in copy_op_keys:
        setattr(tempop, k, getattr(_self.op, k))
    return tempop(x)


class ApplyRAUNet:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "input_blocks": ("STRING", {"default": "3"}),
                "output_blocks": ("STRING", {"default": "8"}),
                "time_mode": (
                    (
                        "percent",
                        "timestep",
                        "sigma",
                    ),
                ),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "end_time": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 999.0}),
                "upscale_mode": (UPSCALE_METHODS,),
                "ca_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "ca_end_time": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 999.0}),
                "ca_input_blocks": ("STRING", {"default": "4"}),
                "ca_output_blocks": ("STRING", {"default": "8"}),
                "ca_upscale_mode": (UPSCALE_METHODS,),
                "two_stage_upscale_mode": (["disabled", *UPSCALE_METHODS], {"default": "disabled"}),
            },
        }

    def patch(
        self,
        model: ModelPatcher,
        input_blocks,
        output_blocks,
        time_mode,
        start_time,
        end_time,
        upscale_mode,
        ca_start_time,
        ca_end_time,
        ca_input_blocks,
        ca_output_blocks,
        ca_upscale_mode,
        two_stage_upscale_mode,
    ):
        use_blocks = parse_blocks("input", input_blocks)
        use_blocks |= parse_blocks("output", output_blocks)
        ca_use_blocks = parse_blocks("input", ca_input_blocks)
        ca_use_blocks |= parse_blocks("output", ca_output_blocks)

        model = model.clone()

        ms = model.get_model_object("model_sampling")

        start_sigma, end_sigma = convert_time(
            ms,
            time_mode,
            start_time,
            end_time,
        )
        ca_start_sigma, ca_end_sigma = convert_time(
            ms,
            time_mode,
            ca_start_time,
            ca_end_time,
        )

        def input_block_patch(h, extra_options):
            if extra_options.get("block") not in ca_use_blocks or not check_time(
                extra_options,
                ca_start_sigma,
                ca_end_sigma,
            ):
                return h
            return F.avg_pool2d(h, kernel_size=(2, 2))

        def output_block_patch(h, hsp, extra_options):
            if extra_options.get("block") not in ca_use_blocks or not check_time(
                extra_options,
                ca_start_sigma,
                ca_end_sigma,
            ):
                return h, hsp
            sigma = get_sigma(extra_options)
            block = extra_options.get("block", ("", 0))[1]
            if sigma is not None and (block < 3 or block > 6):
                sigma /= 16
            return scale_samples(
                h,
                hsp.shape[3],
                hsp.shape[2],
                mode=ca_upscale_mode,
                sigma=sigma,
            ), hsp

        hdconfig = HDConfig(start_sigma, end_sigma, use_blocks, upscale_mode, two_stage_upscale_mode)

        model.set_model_input_block_patch(input_block_patch)
        model.set_model_output_block_patch(output_block_patch)

        for block_type, block_index in use_blocks:
            blocks = getattr(model.model.diffusion_model, f"{block_type}_blocks")
            block_name = f"hd_{block_type}_{block_index}"

            if block_type == "input":
                block = blocks[block_index][0]
                model.add_object_patch(f"{block_name}.forward", partial(forward_downsample, block, block.forward, hdconfig))
                setattr(model.model, block_name, block)
            elif block_type == "output":
                block = blocks[block_index][2]
                model.add_object_patch(f"{block_name}.forward", partial(forward_upsample, block, block.forward, hdconfig))
                setattr(model.model, block_name, block)

        try_patch_forward_timestep_embed()
        try_patch_apply_control()
        try_patch_freeu_advanced()

        return (model,)


class ApplyRAUNetSimple:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (("SD15", "SDXL"),),
                "res_mode": (
                    (
                        "high (1536-2048)",
                        "low (1024 or lower)",
                        "ultra (over 2048)",
                    ),
                ),
                "upscale_mode": (
                    (
                        "default",
                        *UPSCALE_METHODS,
                    ),
                ),
                "ca_upscale_mode": (
                    (
                        "default",
                        *UPSCALE_METHODS,
                    ),
                ),
                "two_stage_upscale_mode": (
                    (
                        "default",
                        "disabled",
                        *UPSCALE_METHODS,
                    ),
                ),
            },
        }

    def patch(self, model_type, res_mode, upscale_mode, ca_upscale_mode, two_stage_upscale_mode, model):
        if upscale_mode == "default":
            upscale_mode = "bicubic"
        if ca_upscale_mode == "default":
            ca_upscale_mode = "bicubic"
        if two_stage_upscale_mode == "default":
            two_stage_upscale_mode = "disabled"
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

        prettyblocks = " / ".join(b if b else "none" for b in blocks)
        prettycablocks = " / ".join(b if b else "none" for b in ca_blocks)
        logging.info(
            f"** ApplyRAUNetSimple: Using preset {model_type} {res}: upscale {upscale_mode}, in/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}  |  CA upscale {ca_upscale_mode},  CA in/out blocks [{prettycablocks}], CA start/end percent {ca_time_range[0]:.2}/{ca_time_range[1]:.2}",
        )
        return ApplyRAUNet().patch(
            model,
            *blocks,
            "percent",
            *time_range,
            upscale_mode,
            *ca_time_range,
            *ca_blocks,
            ca_upscale_mode,
            two_stage_upscale_mode,
        )


__all__ = ("ApplyRAUNet", "ApplyRAUNetSimple")
