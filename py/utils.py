from __future__ import annotations

import importlib
import math
from enum import StrEnum
from typing import Sequence

import torch.nn.functional as torchf
from comfy import latent_formats
from comfy.utils import bislerp

UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "nearest", "area")


class TimeMode(StrEnum):
    PERCENT = "percent"
    TIMESTEP = "timestep"
    SIGMA = "sigma"


class ModelType(StrEnum):
    SD15 = "SD15"
    SDXL = "SDXL"


def parse_blocks(name: str, val: str | Sequence[int]) -> set[tuple[str, int]]:
    if isinstance(val, (tuple, list)):
        # Handle a sequence passed in via YAML parameters.
        if not all(isinstance(item, int) and item >= 0 for item in val):
            raise ValueError(
                "Bad blocks definition, must be comma separated string or sequence of positive int",
            )
        return {(name, item) for item in val}
    vals = (rawval.strip() for rawval in val.split(","))
    return {(name, int(val.strip())) for val in vals if val}


def convert_time(
    ms: object,
    time_mode: TimeMode,
    start_time: float,
    end_time: float,
) -> tuple[float, float]:
    if time_mode == TimeMode.SIGMA:
        return (start_time, end_time)
    if time_mode == TimeMode.TIMESTEP:
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
    return (
        round(ms.percent_to_sigma(start_time), 4),
        round(ms.percent_to_sigma(end_time), 4),
    )
    raise ValueError("invalid time mode")


def get_sigma(options: dict, key: str = "sigmas") -> None | float:
    if not isinstance(options, dict):
        return None
    sigmas = options.get(key)
    if sigmas is None:
        return None
    if isinstance(sigmas, float):
        return sigmas
    return sigmas.detach().cpu().max().item()


def check_time(time_arg: dict | float, start_sigma: float, end_sigma: float) -> bool:
    sigma = get_sigma(time_arg) if not isinstance(time_arg, float) else time_arg
    if sigma is None:
        return False
    return sigma <= start_sigma and sigma >= end_sigma


__block_to_num_map = {"input": 0, "middle": 1, "output": 2}


def block_to_num(block_type: str, block_id: int) -> tuple[int, int]:
    type_id = __block_to_num_map.get(block_type)
    if type_id is None:
        errstr = f"Got unexpected block type {block_type}!"
        raise ValueError(errstr)
    return (type_id, block_id)


# Naive and totally inaccurate way to factorize target_res into rescaled integer width/height
def rescale_size(width: int, height: int, target_res: int) -> tuple[int, int]:
    def get_neighbors(num: float) -> set:
        def f_c(a: float) -> tuple[int, int]:
            return (math.floor(a), math.ceil(a))

        return {*f_c(num - 1), *f_c(num), *f_c(num + 1)}

    scale = math.sqrt(height * width / target_res)
    height_scaled, width_scaled = height / scale, width / scale
    height_rounded = get_neighbors(height_scaled)
    width_rounded = get_neighbors(width_scaled)

    for w in width_rounded:
        h_ = target_res / w
        if h_ % 1 == 0:
            return w, int(h_)
    for h in height_rounded:
        w_ = target_res / h
        if w_ % 1 == 0:
            return int(w_), h

    msg = f"Can't rescale {width} and {height} to fit {target_res}"
    raise ValueError(msg)


def guess_model_type(model: object) -> None | ModelType:
    latent_format = model.get_model_object("latent_format")
    if isinstance(latent_format, latent_formats.SD15):
        return ModelType.SD15
    if isinstance(
        latent_format,
        (latent_formats.SDXL, latent_formats.SDXL_Playground_2_5),
    ):
        return ModelType.SDXL
    return None


def sigma_to_pct(ms, sigma):
    return (1.0 - (ms.timestep(sigma).detach().cpu() / 999.0)).clamp(0.0, 1.0).item()


def fade_scale(
    pct,
    start_pct=0.0,
    end_pct=1.0,
    fade_start=1.0,
    fade_cap=0.0,
):
    if not (start_pct <= pct <= end_pct) or start_pct > end_pct:
        return 0.0
    if pct < fade_start:
        return 1.0
    scaling_pct = 1.0 - ((pct - fade_start) / (end_pct - fade_start))
    return max(fade_cap, scaling_pct)


try:
    bleh = importlib.import_module("custom_nodes.ComfyUI-bleh")
    bleh_latentutils = getattr(bleh.py, "latent_utils", None)
    if bleh_latentutils is None:
        raise ImportError  # noqa: TRY301
    bleh_version = getattr(bleh, "BLEH_VERSION", -1)
    if bleh_version < 0:

        def scale_samples(*args: list, sigma=None, **kwargs: dict):  # noqa: ARG001
            return bleh_latentutils.scale_samples(*args, **kwargs)

    else:
        scale_samples = bleh_latentutils.scale_samples
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
except (ImportError, NotImplementedError):

    def scale_samples(
        samples,
        width,
        height,
        mode="bicubic",
        sigma=None,  # noqa: ARG001
    ):
        if mode == "bislerp":
            return bislerp(samples, width, height)
        return torchf.interpolate(samples, size=(height, width), mode=mode)


__all__ = (
    "UPSCALE_METHODS",
    "check_time",
    "convert_time",
    "get_sigma",
    "guess_model_type",
    "parse_blocks",
    "rescale_size",
    "scale_samples",
)
