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
                "input_blocks": ("STRING", {"default": "0,1"}),
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

        window_args = None
        last_shift = None

        model = model.clone()
        ms = model.get_model_object("model_sampling")

        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)

        def attn1_patch(n, context_attn1, value_attn1, extra_options):
            nonlocal window_args, last_shift
            window_args = None
            if extra_options.get("block") not in use_blocks or not check_time(
                extra_options.get("sigmas"),
                start_sigma,
                end_sigma,
            ):
                return n, context_attn1, value_attn1

            # MSW-MSA
            batch, features, channels = n.shape
            orig_height, orig_width = extra_options["original_shape"][-2:]

            downsample_ratio = int(
                ((orig_height * orig_width) // features) ** 0.5,
            )
            height, width = (
                orig_height // downsample_ratio,
                orig_width // downsample_ratio,
            )
            window_size = (height // 2, width // 2)

            curr_shift = int(torch.rand(1, device="cpu").item() * 4)
            if curr_shift == last_shift:
                curr_shift = (curr_shift + 1) % 4
            last_shift = curr_shift
            match curr_shift:
                case 0:
                    shift_size = (0, 0)
                case 1:
                    shift_size = (window_size[0] // 4, window_size[1] // 4)
                case 2:
                    shift_size = (window_size[0] // 4 * 2, window_size[1] // 4 * 2)
                case _:
                    shift_size = (window_size[0] // 4 * 3, window_size[1] // 4 * 3)
            window_args = (window_size, shift_size, height, width)
            result = self.window_partition(n, *window_args)
            return (result, None, None)

        def attn1_output_patch(n, _extra_options):
            nonlocal window_args
            if window_args is None:
                return n
            result = self.window_reverse(n, *window_args)
            window_args = None
            return result

        model.set_model_attn1_patch(attn1_patch)
        model.set_model_attn1_output_patch(attn1_output_patch)
        return (model,)
