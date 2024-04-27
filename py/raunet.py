import sys

import torch
from comfy.ldm.modules.diffusionmodules import openaimodel
from comfy.ops import disable_weight_init
from comfy.utils import bislerp

nn = torch.nn
F = nn.functional


class HDConfigClass:
    start_sigma = None
    end_sigma = None
    use_blocks = None
    two_stage_upscale = True
    upscale_mode = "bislerp"

    def check(self, topts):
        if topts is None:
            return False
        sigma = topts.get("sigmas")
        if sigma is None:
            return False
        block = topts.get("block")
        sigma = sigma.max().item()
        if self.start_sigma is None or sigma > self.start_sigma:
            return False
        if self.end_sigma is None or sigma <= self.end_sigma:
            return False
        return self.use_blocks and block in self.use_blocks


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
        if HDCONFIG.start_sigma is None:
            return ORIG_FORWARD_TIMESTEP_EMBED(*args, **kwargs)
        return hd_forward_timestep_embed(*args, **kwargs)

    fua_nodes.forward_timestep_embed = fwd_ts_embed
    print("** Patched FreeU_Advanced")


def hd_forward_timestep_embed(ts, x, emb, *args: list, **kwargs: dict):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        # May have been passed positionally.
        transformer_options = args[1] if args else {}
    output_shape = kwargs.get("output_shape")
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
        if output_shape is not None:
            shape = (output_shape[2], output_shape[3])
        else:
            shape = (x.shape[2] * 2, x.shape[3] * 2)
        if HDCONFIG.two_stage_upscale:
            x = F.interpolate(x, size=shape, mode="nearest")
        if HDCONFIG.upscale_mode == "bislerp":
            x = bislerp(x, shape[1] * 2, shape[0] * 2)
        else:
            x = F.interpolate(
                x,
                size=(shape[0] * 2, shape[1] * 2),
                mode=HDCONFIG.upscale_mode,
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
                "upscale_mode": (
                    ("bicubic", "bislerp", "bilinear", "nearest-exact", "area"),
                ),
                "ca_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "ca_end_time": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 999.0}),
                "ca_input_blocks": ("STRING", {"default": "4"}),
                "ca_output_blocks": ("STRING", {"default": "8"}),
                "ca_upscale_mode": (
                    ("bicubic", "bislerp", "bilinear", "nearest-exact", "area"),
                ),
                "model": ("MODEL",),
            },
        }

    @staticmethod
    def parse_blocks(name, s) -> set:
        vals = (rawval.strip() for rawval in s.split(","))
        return {(name, int(val.strip())) for val in vals if val}

    @staticmethod
    def convert_time(ms, time_mode, start_time, end_time) -> tuple:
        match time_mode:
            case "sigma":
                return (start_time, end_time)
            case "percent" | "timestep":
                if time_mode == "timestep":
                    start_time = 1.0 - (start_time / 999.0)
                    end_time = 1.0 - (end_time / 999.0)
                else:
                    if start_time > 1.0 or start_time < 0.0:
                        raise ValueError(
                            "invalid value for start percent",
                        )
                    if end_time > 1.0 or end_time < 0.0:
                        raise ValueError(
                            "invalid value for end percent",
                        )
                return (ms.percent_to_sigma(start_time), ms.percent_to_sigma(end_time))
            case _:
                raise ValueError("invalid time mode")

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
        use_blocks = self.parse_blocks("input", input_blocks)
        use_blocks |= self.parse_blocks("output", output_blocks)
        ca_use_blocks = self.parse_blocks("input", ca_input_blocks)
        ca_use_blocks |= self.parse_blocks("output", ca_output_blocks)

        model = model.clone()
        if not enabled:
            HDCONFIG.start_sigma = HDCONFIG.end_sigma = None
            if ORIG_FORWARD_TIMESTEP_EMBED is not None:
                openaimodel.forward_timestep_embed = ORIG_FORWARD_TIMESTEP_EMBED
            return (model,)

        ms = model.get_model_object("model_sampling")

        HDCONFIG.start_sigma, HDCONFIG.end_sigma = self.convert_time(
            ms,
            time_mode,
            start_time,
            end_time,
        )
        ca_start_sigma, ca_end_sigma = self.convert_time(
            ms,
            time_mode,
            ca_start_time,
            ca_end_time,
        )

        def input_block_patch(h, extra_options):
            sigma = extra_options["sigmas"].max().item()
            block = extra_options["block"]
            if (
                block not in ca_use_blocks
                or sigma > ca_start_sigma
                or sigma <= ca_end_sigma
            ):
                return h
            return F.avg_pool2d(h, kernel_size=(2, 2))

        def output_block_patch(h, hsp, extra_options):
            sigma = extra_options["sigmas"].max().item()
            block = extra_options["block"]
            if (
                block not in ca_use_blocks
                or sigma > ca_start_sigma
                or sigma <= ca_end_sigma
                or h.shape[2:4] == hsp.shape[2:4]
            ):
                return h, hsp
            if ca_upscale_mode == "bislerp":
                return bislerp(h, hsp.shape[3], hsp.shape[2]), hsp
            return F.interpolate(
                h,
                size=(hsp.shape[2], hsp.shape[3]),
                mode=ca_upscale_mode,
            ), hsp

        model.set_model_input_block_patch(input_block_patch)
        model.set_model_output_block_patch(output_block_patch)
        HDCONFIG.use_blocks = use_blocks
        HDCONFIG.two_stage = two_stage_upscale
        HDCONFIG.upscale_mode = upscale_mode
        if openaimodel.forward_timestep_embed is not hd_forward_timestep_embed:
            try_patch_freeu_advanced()
            ORIG_FORWARD_TIMESTEP_EMBED = openaimodel.forward_timestep_embed
            openaimodel.forward_timestep_embed = hd_forward_timestep_embed
        return (model,)
