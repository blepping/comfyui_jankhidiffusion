import importlib

import torch.nn.functional as torchf
from comfy.utils import bislerp

UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "area")


def parse_blocks(name, s) -> set:
    vals = (rawval.strip() for rawval in s.split(","))
    return {(name, int(val.strip())) for val in vals if val}


def convert_time(ms, time_mode, start_time, end_time) -> tuple:
    if time_mode == "sigma":
        return (start_time, end_time)
    if time_mode in ("percent", "timestep"):
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
    raise ValueError("invalid time mode")


def get_sigma(options, key="sigmas"):
    if not isinstance(options, dict):
        return None
    sigmas = options.get(key)
    if sigmas is None:
        return None
    return sigmas.detach().cpu().max().item()


def check_time(options, start_sigma, end_sigma):
    sigma = get_sigma(options)
    if sigma is None:
        return False
    return sigma <= start_sigma and sigma >= end_sigma


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
    "check_time",
    "convert_time",
    "get_sigma",
    "parse_blocks",
    "scale_samples",
    "UPSCALE_METHODS",
)
