import os
import sys

import torch
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ops import disable_weight_init

from .utils import *

nn = torch.nn
F = nn.functional

NO_CONTROLNET_WORKAROUND = (
    os.environ.get("JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND") is not None
)
CONTROLNET_SCALE_ARGS = {"mode": "bilinear", "align_corners": False}


class HDConfigClass:
    enabled = False
    start_sigma = None
    end_sigma = None
    use_blocks = None
    two_stage_upscale = True
    upscale_mode = "bislerp"

    def check(self, topts):
        if not self.enabled or not isinstance(topts, dict):
            return False
        if topts.get("block") not in self.use_blocks:
            return False
        return check_time(topts, self.start_sigma, self.end_sigma)


HDCONFIG = HDConfigClass()

ORIG_FORWARD_TIMESTEP_EMBED = openaimodel.forward_timestep_embed
ORIG_APPLY_CONTROL = openaimodel.apply_control

PATCHED_FREEU = False


# Try to be compatible with FreeU Advanced.
def try_patch_freeu_advanced():
    global PATCHED_FREEU  # noqa: PLW0603
    if PATCHED_FREEU:
        return
    # We only try one time.
    PATCHED_FREEU = True
    fua_nodes = sys.modules.get("FreeU_Advanced.nodes")
    if not fua_nodes:
        return

    def fu_forward_timestep_embed(*args: list, **kwargs: dict):
        fun = (
            hd_forward_timestep_embed
            if HDCONFIG.enabled
            else ORIG_FORWARD_TIMESTEP_EMBED
        )
        return fun(*args, **kwargs)
        if not HDCONFIG.enabled:
            return ORIG_FORWARD_TIMESTEP_EMBED(*args, **kwargs)
        return hd_forward_timestep_embed(*args, **kwargs)

    def fu_apply_control(*args: list, **kwargs: dict):
        fun = hd_apply_control if HDCONFIG.enabled else ORIG_APPLY_CONTROL
        return fun(*args, **kwargs)

    fua_nodes.forward_timestep_embed = fu_forward_timestep_embed
    if not NO_CONTROLNET_WORKAROUND:
        fua_nodes.apply_control = fu_apply_control
    print("** jankhidiffusion: Patched FreeU_Advanced")


def hd_apply_control(h, control, name):
    ctrls = control.get(name) if control is not None else None
    if ctrls is None or len(ctrls) == 0:
        return h
    ctrl = ctrls.pop()
    if ctrl is None:
        return h
    if ctrl.shape[-2:] != h.shape[-2:]:
        print(
            f"* jankhidiffusion: Scaling controlnet conditioning: {ctrl.shape[-2:]} -> {h.shape[-2:]}",
        )
        ctrl = F.interpolate(ctrl, size=h.shape[-2:], **CONTROLNET_SCALE_ARGS)
    h += ctrl
    return h


def try_patch_apply_control():
    global ORIG_APPLY_CONTROL  # noqa: PLW0603
    if openaimodel.apply_control is hd_apply_control or NO_CONTROLNET_WORKAROUND:
        return
    ORIG_APPLY_CONTROL = openaimodel.apply_control
    openaimodel.apply_control = hd_apply_control


class NotFound:
    pass


def hd_forward_timestep_embed(ts, x, emb, *args: list, **kwargs: dict):
    transformer_options = kwargs.get("transformer_options", NotFound)
    output_shape = kwargs.get("output_shape", NotFound)
    transformer_options = (
        args[1] if transformer_options is NotFound and len(args) > 1 else {}
    )
    output_shape = args[2] if output_shape is NotFound and len(args) > 2 else None
    for layer in ts:
        if isinstance(layer, HDUpsample):
            x = layer.forward(
                x,
                output_shape=output_shape,
                transformer_options=transformer_options,
            )
        elif isinstance(layer, HDDownsample):
            x = layer.forward(x, transformer_options=transformer_options)
        else:
            x = ORIG_FORWARD_TIMESTEP_EMBED((layer,), x, emb, *args, **kwargs)
    return x


OrigUpsample, OrigDownsample = openaimodel.Upsample, openaimodel.Downsample


class HDUpsample(OrigUpsample):
    def forward(self, x, output_shape=None, transformer_options=None):
        if (
            self.dims == 3
            or not self.use_conv
            or not HDCONFIG.check(transformer_options)
        ):
            return super().forward(x, output_shape=output_shape)
        shape = (
            output_shape[2:4]
            if output_shape is not None
            else (x.shape[2] * 4, x.shape[3] * 4)
        )
        if HDCONFIG.two_stage_upscale:
            x = F.interpolate(x, size=(shape[0] // 2, shape[1] // 2), mode="nearest")
        x = scale_samples(
            x,
            shape[1],
            shape[0],
            mode=HDCONFIG.upscale_mode,
            sigma=get_sigma(transformer_options),
        )
        return self.conv(x)


class HDDownsample(OrigDownsample):
    COPY_OP_KEYS = (
        "comfy_cast_weights",
        "weight_function",
        "bias_function",
        "weight",
        "bias",
    )

    def __init__(self, *args: list, dtype=None, device=None, **kwargs: dict):
        super().__init__(*args, dtype=dtype, device=device, **kwargs)
        self.dtype = dtype
        self.device = device
        self.ops = kwargs.get("operations", disable_weight_init)

    def forward(self, x, transformer_options=None):
        if (
            self.dims == 3
            or not self.use_conv
            or not HDCONFIG.check(transformer_options)
        ):
            return super().forward(x)
        tempop = self.ops.conv_nd(
            self.dims,
            self.channels,
            self.out_channels,
            3,  # kernel size
            stride=(4, 4),
            padding=(2, 2),
            dilation=(2, 2),
            dtype=self.dtype,
            device=self.device,
        )
        for k in self.COPY_OP_KEYS:
            setattr(tempop, k, getattr(self.op, k))
        return tempop(x)


# Necessary to monkeypatch the built in blocks before any models are loaded.
openaimodel.Upsample = HDUpsample
openaimodel.Downsample = HDDownsample


class ApplyRAUNet:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
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
                "two_stage_upscale": ("BOOLEAN", {"default": False}),
                "upscale_mode": (UPSCALE_METHODS,),
                "ca_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "ca_end_time": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 999.0}),
                "ca_input_blocks": ("STRING", {"default": "4"}),
                "ca_output_blocks": ("STRING", {"default": "8"}),
                "ca_upscale_mode": (UPSCALE_METHODS,),
                "model": ("MODEL",),
            },
        }

    def patch(
        self,
        enabled,
        model,
        input_blocks,
        output_blocks,
        time_mode,
        start_time,
        end_time,
        two_stage_upscale,
        upscale_mode,
        ca_start_time,
        ca_end_time,
        ca_input_blocks,
        ca_output_blocks,
        ca_upscale_mode,
    ):
        global ORIG_FORWARD_TIMESTEP_EMBED  # noqa: PLW0603
        use_blocks = parse_blocks("input", input_blocks)
        use_blocks |= parse_blocks("output", output_blocks)
        ca_use_blocks = parse_blocks("input", ca_input_blocks)
        ca_use_blocks |= parse_blocks("output", ca_output_blocks)

        model = model.clone()
        if not enabled:
            HDCONFIG.enabled = False
            if ORIG_FORWARD_TIMESTEP_EMBED is not None:
                openaimodel.forward_timestep_embed = ORIG_FORWARD_TIMESTEP_EMBED
            if (
                openaimodel.apply_control is not ORIG_APPLY_CONTROL
                and not NO_CONTROLNET_WORKAROUND
            ):
                openaimodel.apply_control = ORIG_APPLY_CONTROL
            return (model,)

        ms = model.get_model_object("model_sampling")

        HDCONFIG.start_sigma, HDCONFIG.end_sigma = convert_time(
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

        model.set_model_input_block_patch(input_block_patch)
        model.set_model_output_block_patch(output_block_patch)
        HDCONFIG.use_blocks = use_blocks
        HDCONFIG.two_stage_upscale = two_stage_upscale
        HDCONFIG.upscale_mode = upscale_mode
        HDCONFIG.enabled = True
        if openaimodel.forward_timestep_embed is not hd_forward_timestep_embed:
            try_patch_freeu_advanced()
            ORIG_FORWARD_TIMESTEP_EMBED = openaimodel.forward_timestep_embed
            openaimodel.forward_timestep_embed = hd_forward_timestep_embed
        try_patch_apply_control()
        return (model,)


class ApplyRAUNetSimple:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "go"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
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
                "model": ("MODEL",),
            },
        }

    def go(self, enabled, model_type, res_mode, upscale_mode, ca_upscale_mode, model):
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
                enabled = False
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
        if not enabled:
            print("** ApplyRAUNetSimple: Disabled")
            return (model.clone(),)
        prettyblocks = " / ".join(b if b else "none" for b in blocks)
        prettycablocks = " / ".join(b if b else "none" for b in ca_blocks)
        print(
            f"** ApplyRAUNetSimple: Using preset {model_type} {res}: upscale {upscale_mode}, in/out blocks [{prettyblocks}], start/end percent {time_range[0]:.2}/{time_range[1]:.2}  |  CA upscale {ca_upscale_mode},  CA in/out blocks [{prettycablocks}], CA start/end percent {ca_time_range[0]:.2}/{ca_time_range[1]:.2}",
        )
        return ApplyRAUNet().patch(
            True,  # noqa: FBT003
            model,
            *blocks,
            "percent",
            *time_range,
            False,  # noqa: FBT003
            upscale_mode,
            *ca_time_range,
            *ca_blocks,
            ca_upscale_mode,
        )


__all__ = ("ApplyRAUNet", "ApplyRAUNetSimple")
