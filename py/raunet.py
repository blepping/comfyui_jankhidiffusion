import sys

import torch
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ops import disable_weight_init

from .utils import *

nn = torch.nn
F = nn.functional


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

    def fwd_ts_embed(*args: list, **kwargs: dict):
        if not HDCONFIG.enabled:
            return ORIG_FORWARD_TIMESTEP_EMBED(*args, **kwargs)
        return hd_forward_timestep_embed(*args, **kwargs)

    fua_nodes.forward_timestep_embed = fwd_ts_embed
    print("** jankhidiffusion: Patched FreeU_Advanced")


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
                "two_stage_upscale": ("BOOLEAN", {"default": True}),
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
        HDCONFIG.two_stage = two_stage_upscale
        HDCONFIG.upscale_mode = upscale_mode
        HDCONFIG.enabled = True
        if openaimodel.forward_timestep_embed is not hd_forward_timestep_embed:
            try_patch_freeu_advanced()
            ORIG_FORWARD_TIMESTEP_EMBED = openaimodel.forward_timestep_embed
            openaimodel.forward_timestep_embed = hd_forward_timestep_embed
        return (model,)
