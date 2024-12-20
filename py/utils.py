from __future__ import annotations

import contextlib
import importlib
import itertools
import logging
import math
import sys
from typing import TYPE_CHECKING

import torch.nn.functional as torchf
from comfy import latent_formats
from comfy.utils import bislerp

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from enum import StrEnum
except ImportError:
    # Compatibility workaround for pre-3.11 Python versions.
    from enum import Enum

    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(name: str, *_unused: list) -> str:
            return name.lower()

        def __str__(self) -> str:
            return str(self.value)


logger = logging.getLogger(__name__)

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


def get_sigma(options: dict, key: str = "sigmas") -> float | None:
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
def rescale_size(
    width: int,
    height: int,
    target_res: int,
    *,
    tolerance=1,
) -> tuple[int, int]:
    tolerance = min(target_res, tolerance)

    def get_neighbors(num: float):
        if num < 1:
            return None
        numi = int(num)
        return tuple(
            numi + adj
            for adj in sorted(
                range(
                    -min(numi - 1, tolerance),
                    tolerance + 1 + math.ceil(num - numi),
                ),
                key=abs,
            )
        )

    scale = math.sqrt(height * width / target_res)
    height_scaled, width_scaled = height / scale, width / scale
    height_rounded = get_neighbors(height_scaled)
    width_rounded = get_neighbors(width_scaled)
    for h, w in itertools.zip_longest(height_rounded, width_rounded):
        h_adj = target_res / w if w is not None else 0.1
        if h_adj % 1 == 0:
            return (w, int(h_adj))
        if h is None:
            continue
        w_adj = target_res / h
        if w_adj % 1 == 0:
            return (int(w_adj), h)
    msg = f"Can't rescale {width} and {height} to fit {target_res}"
    raise ValueError(msg)


def guess_model_type(model: object) -> ModelType | None:
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


def get_custom_node(name):
    module_key = f"custom_nodes.{name}"
    try:
        spec = importlib.util.find_spec(module_key)
        if spec is None:
            raise ModuleNotFoundError(module_key)
        module = next(
            v
            for v in sys.modules.copy().values()
            if hasattr(v, "__spec__")
            and v.__spec__ is not None
            and v.__spec__.origin == spec.origin
        )
    except StopIteration:
        raise ModuleNotFoundError(module_key) from None
    return module


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


class Integrations:
    def __init__(self):
        self.initialized = False
        self.modules = {}
        self.init_handlers = []

    def __getitem__(self, key):
        return self.modules[key]

    def __contains__(self, key):
        return key in self.modules

    def __getattr__(self, key):
        return self.modules.get(key)

    def register_init_handler(self, fun):
        self.init_handlers.append(fun)

    def initialize(self) -> None:
        if self.initialized:
            return
        self.initialized = True

        with contextlib.suppress(ModuleNotFoundError, NotImplementedError):
            bleh = get_custom_node("ComfyUI-bleh")
            bleh_version = getattr(bleh, "BLEH_VERSION", -1)
            if bleh_version < 0:
                raise NotImplementedError
            self.modules["bleh"] = bleh

        for init_handler in self.init_handlers:
            init_handler()


MODULES = Integrations()


def init_integrations():
    global scale_samples, UPSCALE_METHODS  # noqa: PLW0603
    ext_bleh = MODULES.bleh
    if ext_bleh is None:
        return
    bleh_latentutils = getattr(ext_bleh.py, "latent_utils", None)
    if bleh_latentutils is None:
        return
    bleh_version = getattr(ext_bleh, "BLEH_VERSION", -1)
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
    if bleh_version >= 0:
        scale_samples = bleh_latentutils.scale_samples
        return

    def scale_samples_wrapped(*args: list, sigma=None, **kwargs: dict):  # noqa: ARG001
        return bleh_latentutils.scale_samples(*args, **kwargs)

    scale_samples = scale_samples_wrapped


MODULES.register_init_handler(init_integrations)

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
