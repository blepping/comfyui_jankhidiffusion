from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from comfy.ldm.modules.diffusionmodules import openaimodel

from . import utils
from .utils import (
    IntegratedNode,
    ModelType,
    TimeMode,
    check_time,
    convert_time,
    fade_scale,
    get_sigma,
    guess_model_type,
    logger,
    parse_blocks,
    scale_samples,
    sigma_to_pct,
)

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

F = torch.nn.functional

CA_DOWNSCALE_METHODS = ()


def init_integrations(_integrations) -> None:
    global scale_samples, CA_DOWNSCALE_METHODS  # noqa: PLW0603
    CA_DOWNSCALE_METHODS = (
        ("avg_pool2d", "adaptive_avg_pool2d", *utils.UPSCALE_METHODS)
        if "adaptive_avg_pool2d" not in utils.UPSCALE_METHODS
        else ("avg_pool2d", *utils.UPSCALE_METHODS)
    )
    scale_samples = utils.scale_samples


utils.MODULES.register_init_handler(init_integrations)


class Preset(NamedTuple):
    input_blocks: str = ""
    output_blocks: str = ""
    time_mode: TimeMode = TimeMode.PERCENT
    start_time: float = 1.0
    end_time: float = 1.0
    upscale_mode: str = "bicubic"
    ca_start_time: float = 1.0
    ca_end_time: float = 1.0
    ca_downscale_factor: float = 2.0
    ca_input_blocks: str = ""
    ca_output_blocks: str = ""
    ca_upscale_mode: str = "bicubic"
    ca_downscale_mode: str = "avg_pool2d"
    ca_input_after_skip_mode: bool = False
    two_stage_upscale_mode: str = "disabled"

    def _pretty_blocks(self, *, ca: bool = False) -> str:
        if ca:
            blocks = (
                self.ca_input_blocks,
                self.ca_output_blocks,
            )
        else:
            blocks = (self.input_blocks, set(), self.output_blocks)
        return " / ".join(b or "none" for b in blocks)

    @property
    def pretty_blocks(self) -> str:
        return self._pretty_blocks(ca=False)

    @property
    def ca_pretty_blocks(self) -> str:
        return self._pretty_blocks(ca=True)

    @property
    def as_dict(self):
        return {k: getattr(self, k) for k in self._fields}

    def edited(self, **kwargs: dict) -> NamedTuple:
        kwargs = self.as_dict | kwargs
        return self.__class__(**kwargs)


SD15_PRESET = Preset(
    input_blocks="3",
    output_blocks="8",
    ca_input_blocks="1",
    ca_output_blocks="11",
)

SDXL_PRESET = Preset(
    input_blocks="3",
    output_blocks="5",
    ca_input_blocks="4",
    ca_output_blocks="5",
)

SIMPLE_PRESETS = {
    "SD15_low": SD15_PRESET.edited(
        start_time=0.0,
        end_time=0.4,
    ),
    "SD15_high": SD15_PRESET.edited(
        start_time=0.0,
        end_time=0.5,
        ca_start_time=0.0,
        ca_end_time=0.35,
    ),
    "SD15_ultra": SD15_PRESET.edited(
        start_time=0.0,
        end_time=0.6,
        ca_start_time=0.0,
        ca_end_time=0.45,
    ),
    "SDXL_low": SDXL_PRESET.edited(),  # ???
    "SDXL_high": SDXL_PRESET.edited(
        ca_start_time=0.0,
        ca_end_time=0.5,
    ),
    "SDXL_ultra": SDXL_PRESET.edited(
        start_time=0.0,
        end_time=0.45,
        ca_start_time=0.0,
        ca_end_time=0.6,
    ),
}


@dataclass
class Config:
    start_sigma: float
    end_sigma: float
    ca_start_sigma: float
    ca_end_sigma: float
    use_blocks: set
    ca_use_blocks: set
    upscale_mode: str = "bicubic"
    two_stage_upscale_mode: str = "disabled"
    ca_upscale_mode: str = "bicubic"
    ca_downscale_mode: str = "adaptive_avg_pool2d"
    ca_downscale_factor: float = 2.0
    ca_downscale_factor_w: float | None = None
    # Patches the input  block after the skip connection.
    ca_input_after_skip_mode: bool = False
    ca_avg_pool2d_ceil_mode: bool = True
    # Hack for ComfyUI-bleh latent effects.  # noqa: FIX004
    ca_output_sigma_hack: bool = True
    # Scaling on the tensors going in/out of scaling.
    pre_upscale_multiplier: float = 1.0
    post_upscale_multiplier: float = 1.0
    pre_downscale_multiplier: float = 1.0
    post_downscale_multiplier: float = 1.0
    ca_pre_upscale_multiplier: float = 1.0
    ca_post_upscale_multiplier: float = 1.0
    ca_pre_downscale_multiplier: float = 1.0
    ca_post_downscale_multiplier: float = 1.0
    # Allows fading out the scale effect starting from this time.
    ca_fadeout_start_sigma: float | None = None
    # Maximum fadeout, as a percentage of the total scale effect.
    ca_fadeout_cap: float = 0.0
    ca_latent_pixel_increment: int | float = 8
    verbose: int = 0
    curr_sigma: float | None = None

    @classmethod
    def build(
        cls,
        ms: object,
        *,
        input_blocks: str | list[int],
        output_blocks: str | list[int],
        time_mode: str | TimeMode,
        start_time: float,
        end_time: float,
        ca_start_time: float,
        ca_end_time: float,
        ca_input_blocks: str | list[int],
        ca_output_blocks: str | list[int],
        ca_fadeout_start_time: float | None = None,
        **kwargs: dict,
    ) -> object:
        time_mode: TimeMode = TimeMode(time_mode)
        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)
        ca_start_sigma, ca_end_sigma = convert_time(
            ms,
            time_mode,
            ca_start_time,
            ca_end_time,
        )
        if ca_fadeout_start_time is not None:
            ca_fadeout_start_sigma = convert_time(
                ms,
                time_mode,
                ca_fadeout_start_time,
                ca_fadeout_start_time,
            )[0]
        else:
            ca_fadeout_start_sigma = None
        input_blocks, output_blocks = itertools.starmap(
            parse_blocks,
            (
                ("input", input_blocks),
                ("output", output_blocks),
            ),
        )
        ca_input_blocks, ca_output_blocks = itertools.starmap(
            parse_blocks,
            (
                ("input", ca_input_blocks),
                ("output", ca_output_blocks),
            ),
        )
        return cls(
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            ca_start_sigma=ca_start_sigma,
            ca_end_sigma=ca_end_sigma,
            ca_fadeout_start_sigma=ca_fadeout_start_sigma,
            use_blocks=input_blocks | output_blocks,
            ca_use_blocks=ca_input_blocks | ca_output_blocks,
            **kwargs,
        )

    def check(self, topts: dict, *, ca=False) -> bool:
        start_sigma, end_sigma, use_blocks = (
            (self.ca_start_sigma, self.ca_end_sigma, self.ca_use_blocks)
            if ca
            else (self.start_sigma, self.end_sigma, self.use_blocks)
        )
        if not isinstance(topts, dict) or topts.get("block") not in use_blocks:
            return False
        return check_time(topts, start_sigma, end_sigma)

    @staticmethod
    def maybe_multiply(
        t: torch.Tensor,
        multiplier: float = 1.0,
        post: bool = False,
    ) -> torch.Tensor:
        if multiplier == 1.0:
            return t
        return t.mul_(multiplier) if post else t * multiplier


class State:
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
        control: dict | None,
        name: str,
    ) -> torch.Tensor:
        ctrls = control.get(name) if control is not None else None
        if ctrls is None or len(ctrls) == 0:
            return h
        ctrl = ctrls.pop()
        if ctrl is None:
            return h
        if ctrl.shape[-2:] != h.shape[-2:]:
            logger.info(
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
        logger.info("** jankhidiffusion: Patched openaimodel.apply_control")

    # Try to be compatible with FreeU Advanced.
    def try_patch_freeu_advanced(self) -> None:
        if self.patched_freeu_advanced or self.no_controlnet_workaround:
            return

        # We only try one time.
        self.patched_freeu_advanced = True
        fua_nodes = getattr(utils.MODULES.freeu_advanced, "nodes", None)
        if not fua_nodes:
            return

        self.orig_fua_apply_control = fua_nodes.apply_control
        fua_nodes.apply_control = self.hd_apply_control
        logger.info("** jankhidiffusion: Patched FreeU_Advanced")

    def apply_patches(self) -> None:
        self.try_patch_apply_control()
        self.try_patch_freeu_advanced()

    def revert_patches(self) -> None:
        if openaimodel.apply_control == self.hd_apply_control:
            openaimodel.apply_control = self.orig_apply_control
            logger.info("** jankhidiffusion: Reverted openaimodel.apply_control patch")
        if not self.patched_freeu_advanced:
            return
        fua_nodes = getattr(utils.MODULES.freeu_advanced, "nodes", None)
        if not fua_nodes:
            logger.warning(
                "** jankhidiffusion: Unexpectedly could not revert FreeU_Advanced patches",
            )
            return
        fua_nodes.apply_control = self.orig_fua_apply_control
        self.patched_freeu_advanced = False
        logger.info("** jankhidiffusion: Reverted FreeU_Advanced patch")


GLOBAL_STATE: State = State()


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
        config: Config,
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
        self.config = config
        self.block_index = block_index
        self.forward = self.forward_upsample if is_up else self.forward_downsample

    def __call__(self, *args: list, **kwargs: dict) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward_upsample(
        self,
        x: torch.Tensor,
        output_shape: tuple | None = None,
    ) -> torch.Tensor:
        config = self.config
        orig_block = self.orig_block
        block_index = self.block_index
        if (
            orig_block.dims == 3
            or not orig_block.use_conv
            or not config.check({
                "sigmas": config.curr_sigma,
                "block": ("output", block_index),
            })
        ):
            return self.orig_forward(x, output_shape=output_shape)

        shape = (
            output_shape[2:4]
            if output_shape is not None
            else (x.shape[2] * 4, x.shape[3] * 4)
        )
        x = config.maybe_multiply(x, config.pre_upscale_multiplier)
        if config.two_stage_upscale_mode != "disabled":
            x = scale_samples(
                x,
                shape[1] // 2,
                shape[0] // 2,
                mode=config.two_stage_upscale_mode,
                sigma=config.curr_sigma,
            )
        x = scale_samples(
            x,
            shape[1],
            shape[0],
            mode=config.upscale_mode,
            sigma=config.curr_sigma,
        )
        return config.maybe_multiply(
            orig_block.conv(x),
            config.post_upscale_multiplier,
            post=True,
        )

    def forward_downsample(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        orig_block = self.orig_block
        block_index = self.block_index
        if (
            orig_block.dims == 3
            or not orig_block.use_conv
            or not config.check({
                "sigmas": config.curr_sigma,
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
        x = config.maybe_multiply(x, config.pre_downscale_multiplier)
        if config.pre_downscale_multiplier != 1.0:
            x = x * config.pre_downscale_multiplier
        return config.maybe_multiply(
            tempop(x),
            config.post_downscale_multiplier,
            post=True,
        )


class ApplyRAUNet(metaclass=IntegratedNode):
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
                        "tooltip": "Comma-separated list of input Downsample blocks. Default is for SD 1.5. The corresponding valid block from output_blocks must be set along with input.\nValid blocks for SD1.5: 3, 6, 9\nValid blocks for SDXL: 3, 6. Original Hidiffusion implementation uses 6 for SDXL.",
                    },
                ),
                "output_blocks": (
                    "STRING",
                    {
                        "default": "8",
                        "tooltip": "Comma-separated list of output Upsample blocks. Default is for SD 1.5. The corresponding valid block from input_blocks must be set along with output.\nValid blocks for SD1.5: 8, 5, 2\nValid blocks for SDXL: 5, 2. Original Hidiffusion implementation uses 2 for SDXL.",
                    },
                ),
                "time_mode": (
                    tuple(str(val) for val in TimeMode),
                    {
                        "default": "percent",
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
                    utils.UPSCALE_METHODS,
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
                        "tooltip": "Comma separated list of input cross-attention blocks. Default is for SD1.x, for SDXL you can try using 5 (or just disable it).",
                    },
                ),
                "ca_output_blocks": (
                    "STRING",
                    {
                        "default": "8",
                        "tooltip": "Comma-separated list of output cross-attention blocks. Default is for SD1.x, for SDXL you can try using 4 (or just disable it).",
                    },
                ),
                "ca_upscale_mode": (
                    utils.UPSCALE_METHODS,
                    {
                        "tooltip": "Mode used when upscaling latents in output cross-attention blocks.",
                    },
                ),
                "ca_downscale_mode": (
                    CA_DOWNSCALE_METHODS,
                    {
                        "default": "adaptive_avg_pool2d",
                        "tooltip": "Mode used when downscaling latents in output cross-attention blocks (use avg_pool2d for normal Hidiffusion behavior). adaptive_avg_pool2d should be the same and also supports fractional scales.",
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
                    ("disabled", *utils.UPSCALE_METHODS),
                    {
                        "default": "disabled",
                        "tooltip": "When upscaling in output Upscale blocks (non-NA), do half the upscale with this mode and half with the normal upscale mode. May produce a different effect, isn't necessarily better.",
                    },
                ),
            },
            "optional": {
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. You can also override any of the normal parameters by key. This input can be converted into a multiline text widget. See main README for possible options. Note: When specifying paramaters this way, there is very little error checking.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    @classmethod
    def patch(  # noqa: PLR0914
        cls,
        *,
        model: ModelPatcher,
        yaml_parameters: str | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[ModelPatcher]:
        if yaml_parameters:
            import yaml  # noqa: PLC0415

            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "RAUNet: yaml_parameters must either be null or an object",
                )
            else:
                kwargs |= extra_params
        ms = model.get_model_object("model_sampling")
        config = Config.build(ms, **kwargs)
        if config.ca_downscale_mode == "avg_pool2d" and (
            not config.ca_downscale_factor.is_integer()
            or not (
                config.ca_downscale_factor_w is None
                or config.ca_downscale_factor_w.is_integer()
            )
        ):
            raise ValueError(
                "avg_pool2d downscale mode can only be used with integer downscale factors",
            )
        if config.verbose:
            logger.info(f"** jankhidiffusion: RAUNet: Using config: {config}")
        have_ca_output_blocks = any(bt == "output" for (bt, _) in config.ca_use_blocks)

        model = model.clone()
        downscale_factor = config.ca_downscale_factor
        downscale_factor_w = (
            downscale_factor
            if config.ca_downscale_factor_w is None
            else config.ca_downscale_factor_w
        )
        if config.ca_fadeout_start_sigma is not None:
            ca_start_pct = sigma_to_pct(
                ms,
                torch.tensor(config.ca_start_sigma, dtype=torch.float32),
            )
            ca_end_pct = sigma_to_pct(
                ms,
                torch.tensor(config.ca_end_sigma, dtype=torch.float32),
            )
            ca_fadeout_start_pct = sigma_to_pct(
                ms,
                torch.tensor(config.ca_fadeout_start_sigma, dtype=torch.float32),
            )
        else:
            del ms
            ca_fadeout_start_pct = None

        ca_pixel_increment = max(1, config.ca_latent_pixel_increment)

        def input_block_patch(h: torch.Tensor, extra_options: dict) -> torch.Tensor:
            _block_type, block_index = extra_options.get("block", ("unknown", -1))
            if block_index == 0:
                config.curr_sigma = get_sigma(extra_options)
            if not config.check(extra_options, ca=True):
                return h
            curr_downscale_factor, curr_downscale_factor_w = (
                downscale_factor,
                downscale_factor_w,
            )
            if ca_fadeout_start_pct is not None:
                pct = sigma_to_pct(ms, extra_options["sigmas"].max())
                scale_scale = fade_scale(
                    pct,
                    ca_start_pct,
                    ca_end_pct,
                    ca_fadeout_start_pct,
                    config.ca_fadeout_cap,
                )
                if scale_scale <= 0.0:
                    return h
                if scale_scale < 1.0:
                    curr_downscale_factor = curr_downscale_factor - (
                        curr_downscale_factor - 1.0
                    ) * (1.0 - scale_scale)
                    curr_downscale_factor_w = curr_downscale_factor_w - (
                        curr_downscale_factor_w - 1.0
                    ) * (1.0 - scale_scale)
                # print(
                #     f"\n>>> scale_scale={scale_scale:0.4f}, down=({curr_downscale_factor:0.4f}, {curr_downscale_factor_w:0.4f})",
                # )
            height, width = h.shape[-2:]
            target_h = int(
                max(
                    ca_pixel_increment,
                    ((height / ca_pixel_increment) // curr_downscale_factor)
                    * ca_pixel_increment,
                ),
            )
            target_w = int(
                max(
                    ca_pixel_increment,
                    ((width / ca_pixel_increment) // curr_downscale_factor_w)
                    * ca_pixel_increment,
                ),
            )
            # When downscaling, make sure not to overshoot the original size.
            # When upscaling, don't undershoot the original size.
            target_h = (
                min(height, target_h)
                if curr_downscale_factor >= 1
                else max(height, target_h)
            )
            target_w = (
                min(width, target_w)
                if curr_downscale_factor_w >= 1
                else max(width, target_w)
            )
            if (target_h, target_w) == h.shape[-2:]:
                return h
            h = config.maybe_multiply(h, config.ca_pre_downscale_multiplier)
            if config.ca_downscale_mode == "avg_pool2d":
                return config.maybe_multiply(
                    F.avg_pool2d(
                        h,
                        kernel_size=(
                            max(1, int(height // target_h)),
                            max(1, int(width // target_w)),
                        ),
                        ceil_mode=config.ca_avg_pool2d_ceil_mode,
                    ),
                    config.ca_post_downscale_multiplier,
                    post=True,
                )
            # print(f"\n>> h,w={(height, width)}, targets=({target_h}, {target_w})")
            if config.ca_downscale_mode == "adaptive_avg_pool2d":
                result = F.adaptive_avg_pool2d(h, (target_h, target_w))
            else:
                result = scale_samples(
                    h,
                    target_w,
                    target_h,
                    mode=config.ca_downscale_mode,
                    sigma=config.curr_sigma,
                )
            return config.maybe_multiply(
                result,
                config.ca_post_downscale_multiplier,
                post=True,
            )

        def output_block_patch(
            h: torch.Tensor,
            hsp: torch.Tensor,
            extra_options: dict,
        ) -> torch.Tensor:
            if (
                not config.check(extra_options, ca=True)
                or h.shape[-2:] == hsp.shape[-2:]
            ):
                return h, hsp
            sigma = config.curr_sigma
            block = extra_options.get("block", ("", 0))[1]
            if (
                sigma is not None
                and config.ca_output_sigma_hack
                and (block < 3 or block > 6)
            ):
                sigma /= 16
            h = config.maybe_multiply(h, config.ca_pre_upscale_multiplier)
            return config.maybe_multiply(
                scale_samples(
                    h,
                    hsp.shape[-1],
                    hsp.shape[-2],
                    mode=config.ca_upscale_mode,
                    sigma=sigma,
                ),
                config.ca_post_upscale_multiplier,
                post=True,
            ), hsp

        if config.ca_input_after_skip_mode:
            model.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            model.set_model_input_block_patch(input_block_patch)

        if have_ca_output_blocks:
            model.set_model_output_block_patch(output_block_patch)

        for block_type, block_index in config.use_blocks:
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
                HDForward(block, config, block_index, block_type != "input"),
            )

        GLOBAL_STATE.apply_patches()

        return (model,)


class ApplyRAUNetSimple(metaclass=IntegratedNode):
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
                    ("auto", "SD15", "SDXL"),
                    {
                        "tooltip": "Model type being patched. Generally safe to leave on auto. Choose SD15 for SD 1.4 or SD 2.x.",
                    },
                ),
                "res_mode": (
                    (
                        "high (1536-2048)",
                        "low (1024 or lower)",
                        "ultra (over 2048)",
                    ),
                    {
                        "tooltip": "Resolution mode hint, does not have to correspond to the actual size. Note: Choosing `low` with SDXL simply disables RAUNet as SDXL can natively generate at 1024x1024.",
                    },
                ),
                "upscale_mode": (
                    (
                        "default",
                        *utils.UPSCALE_METHODS,
                    ),
                    {
                        "tooltip": "Method used when upscaling latents in output Upsample blocks.",
                    },
                ),
                "ca_upscale_mode": (
                    (
                        "default",
                        *utils.UPSCALE_METHODS,
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
        model_type: str | ModelType,
        res_mode: str,
        upscale_mode: str,
        ca_upscale_mode: str,
    ) -> tuple[ModelPatcher]:
        if model_type == "auto":
            model_type = guess_model_type(model)
            if model_type not in ModelType:
                raise RuntimeError("Unable to guess model type")
        if upscale_mode == "default":
            upscale_mode = "bicubic"
        if ca_upscale_mode == "default":
            ca_upscale_mode = "bicubic"
        res = res_mode.split(" ", 1)[0]
        preset_key = f"{model_type!s}_{res}"
        preset = SIMPLE_PRESETS.get(preset_key)
        if preset is None:
            errstr = f"Unsupported model_type/res_mode combination {preset_key}"
            raise ValueError(errstr)
        preset = preset.edited(
            upscale_mode=upscale_mode,
            ca_upscale_mode=ca_upscale_mode,
        )
        logger.info(
            f"** ApplyRAUNetSimple: Using preset {model_type!s} {res}: upscale {upscale_mode}, in/out blocks [{preset.pretty_blocks}], start/end percent {preset.start_time:.2}/{preset.end_time:.2}  |  CA upscale {preset.ca_upscale_mode},  CA in/out blocks [{preset.ca_pretty_blocks}], CA start/end percent {preset.ca_start_time:.2}/{preset.ca_end_time:.2}",
        )
        return ApplyRAUNet.patch(model=model, **preset.as_dict)


__all__ = ("ApplyRAUNet", "ApplyRAUNetSimple")
