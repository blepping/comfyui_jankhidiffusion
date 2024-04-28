import importlib

import torch.nn.functional as torchf
from comfy.utils import bislerp

UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "area")


def parse_blocks(name, s) -> set:
    vals = (rawval.strip() for rawval in s.split(","))
    return {(name, int(val.strip())) for val in vals if val}


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


def check_time(sigma, start_sigma, end_sigma):
    if sigma is None:
        return False
    sigma = sigma.detach().cpu().max().item()
    return sigma <= start_sigma and sigma >= end_sigma


try:
    bleh_latentutils = importlib.import_module(
        "custom_nodes.ComfyUI-bleh.py.latent_utils",
    )
    scale_samples = bleh_latentutils.scale_samples
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
except ImportError:

    def scale_samples(
        samples,
        width,
        height,
        mode="bicubic",
    ):
        if mode == "bislerp":
            return bislerp(samples, width, height)
        return torchf.interpolate(samples, size=(height, width), mode=mode)


__all__ = (
    "UPSCALE_METHODS",
    "parse_blocks",
    "convert_time",
    "check_time",
    "scale_samples",
)
