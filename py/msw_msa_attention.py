import torch

from .utils import *


class ApplyMSWMSAAttention:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_blocks": ("STRING", {"default": "1,2"}),
                "middle_blocks": ("STRING", {"default": ""}),
                "output_blocks": ("STRING", {"default": "9,10,11"}),
                "time_mode": (
                    (
                        "percent",
                        "timestep",
                        "sigma",
                    ),
                ),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0}),
                "end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 999.0}),
                "model": ("MODEL",),
            },
        }

    # reference: https://github.com/microsoft/Swin-Transformer
    # Window functions adapted from https://github.com/megvii-research/HiDiffusion
    @staticmethod
    def window_partition(x, window_size, shift_size, height, width) -> torch.Tensor:
        batch, _features, channels = x.shape
        x = x.view(batch, height, width, channels)
        if not isinstance(shift_size, (list, tuple)):
            shift_size = (shift_size, shift_size)
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        x = x.view(
            batch,
            height // window_size[0],
            window_size[0],
            width // window_size[1],
            window_size[1],
            channels,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0], window_size[1], channels)
        )
        return windows.view(-1, window_size[0] * window_size[1], channels)

    @staticmethod
    def window_reverse(windows, window_size, shift_size, height, width) -> torch.Tensor:
        batch, features, channels = windows.shape
        windows = windows.view(-1, window_size[0], window_size[1], channels)
        batch = int(
            windows.shape[0] / (height * width / window_size[0] / window_size[1]),
        )
        x = windows.view(
            batch,
            height // window_size[0],
            width // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
        if not isinstance(shift_size, (list, tuple)):
            shift_size = (shift_size, shift_size)
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        return x.view(batch, height * width, channels)

    @staticmethod
    def get_window_args(n, orig_shape, shift) -> tuple:
        _batch, features, _channels = n.shape
        orig_height, orig_width = orig_shape[-2:]

        downsample_ratio = int(
            ((orig_height * orig_width) // features) ** 0.5,
        )
        height, width = (
            orig_height // downsample_ratio,
            orig_width // downsample_ratio,
        )
        window_size = (height // 2, width // 2)

        match shift:
            case 0:
                shift_size = (0, 0)
            case 1:
                shift_size = (window_size[0] // 4, window_size[1] // 4)
            case 2:
                shift_size = (window_size[0] // 4 * 2, window_size[1] // 4 * 2)
            case _:
                shift_size = (window_size[0] // 4 * 3, window_size[1] // 4 * 3)
        return (window_size, shift_size, height, width)

    def patch(
        self,
        model,
        input_blocks,
        middle_blocks,
        output_blocks,
        time_mode,
        start_time,
        end_time,
    ):
        use_blocks = parse_blocks("input", input_blocks)
        use_blocks |= parse_blocks("middle", middle_blocks)
        use_blocks |= parse_blocks("output", output_blocks)

        window_args = last_block = last_shift = None

        model = model.clone()
        ms = model.get_model_object("model_sampling")

        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)

        def attn1_patch(q, k, v, extra_options):
            nonlocal window_args, last_shift, last_block
            window_args = None
            last_block = extra_options.get("block")
            if last_block not in use_blocks or not check_time(
                extra_options,
                start_sigma,
                end_sigma,
            ):
                return q, k, v
            orig_shape = extra_options["original_shape"]
            # MSW-MSA
            shift = int(torch.rand(1, device="cpu").item() * 4)
            if shift == last_shift:
                shift = (shift + 1) % 4
            last_shift = shift
            window_args = tuple(
                self.get_window_args(x, orig_shape, shift) if x is not None else None
                for x in (q, k, v)
            )
            if q is not None and q is k and q is v:
                return (
                    self.window_partition(
                        q,
                        *window_args[0],
                    ),
                ) * 3
            return tuple(
                self.window_partition(x, *window_args[idx]) if x is not None else None
                for idx, x in enumerate((q, k, v))
            )

        def attn1_output_patch(n, extra_options):
            nonlocal window_args
            if window_args is None or last_block != extra_options.get("block"):
                window_args = None
                return n
            args, window_args = window_args[0], None
            return self.window_reverse(n, *args)

        model.set_model_attn1_patch(attn1_patch)
        model.set_model_attn1_output_patch(attn1_output_patch)
        return (model,)
