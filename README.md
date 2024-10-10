# ComfyUI jank HiDiffusion

Janky experimental implementation of [HiDiffusion](https://github.com/megvii-research/HiDiffusion) for
[ComfyUI](https://github.com/comfyanonymous/ComfyUI).

See the [changelog](changelog.md) for recent user-visible changes.

## Description

Read the link above for an official description. The following is just my understanding and may or may not be
correct:

As far as I understand it, the RAU-Net part is essentially Kohya Deep Shrink (AKA `PatchModelAddDownscale`):
the concept is to scale down the image at the start of generation to let the model set up major details like
how many legs a character has and then allow the model to refine and add detail once the scaling effect ends.
The main difference for that part is the downscale methods - it uses convolution with stride/dilation
and pool averaging to downscale while Deep Shrink usually uses bicubic downscaling. Where the scaling occurs
also may be important — it does seem to work noticeably better than Deep Shrink, at least for SD 1.5.

Not sure how to describe MSW-MSA attention. It seems like a big performance boost for SD 1.5 at high res and also
appears to increase quality. Note that it does not enable high res generation by itself.

## Caveats

I'm not an expert on diffusion stuff and I'm not sure I fully understood the HiDiffusion code. My implementation
may or may not work correctly. If you experience issues, please don't blame HiDiffusion unless you can also reproduce
it with their implementation.

I mainly use SD 1.5 models: generally these nodes should work well with SD 1.5. I've found SDXL in general doesn't
tolerate these Deep Shrink type effects as well as SD 1.5 and it is also less tested. If using SDXL your mileage
may vary.

**Important**: The advanced node default values are for SD 1.5, they won't work well for other models (like SDXL). This
stuff probably doesn't work at all for more exotic models like Cascade.

* Not all aspect ratios work with the MSW-MSA attention node. It may be the same with the original implementation? Try to
  use resolutions that are multiples of 64 or 128.
* The RAUNet component may not work properly with ControlNet while the scaling effect is active.
* The MSW-MSA attention node doesn't seem to help performance with SDXL much.
* I may not have implemented the cross-attention block part correctly. As far as I could tell, it seemed like it
  was just patching a normal block, not actual cross-attention (so almost exactly like Deep Shrink). My version
  is implemented that way and does not use an actual attention patch.
* Customizable, but not very userfriendly. You get to figure out the blocks to patch!
* The list of caveats is too long, and it's probably not even complete. Yikes!

## Use with ControlNet

First: RAUNet is used to help the model construct major details like how many legs a creature has when working at
resolutions above what it was trained on. With ControlNet guidance, you very likely don't need RAUNet and similar
effects. Don't use it unless you actually _need_ to.

If you do use RAUNet and ControlNet concurrently, I recommend adjusting the RAUNet parametrs to only apply the effect
for a short time - the minimum necessary. For example, if you'd normaly use an end time of 0.5 and CA end time of
0.3 then with ControlNet you may want to use an end time of 0.3 and just disable the CA effect entirely. Or apply
it very briefly, something like CA end time 0.15.

I now try to apply a workaround to scale the ControlNet conditioning when the RAUNet effect is active. This is
_probably_ better than nothing but likely still incorrect. When it's working, you'll see messages like this
in your log:

```plaintext
* jankhidiffusion: Scaling controlnet conditioning: torch.Size([24, 24]) -> torch.Size([12, 12])
```

If you find the workaround is causing issues, set the environment variable `JANKHIDIFFUSION_DISABLE_CONTROLNET_WORKAROUND`
to any value.

Ancestral samplers seem to work a lot better than the non-ancestral ones when using RAUNet and ControlNet simultaneously. I
recommend using the ancestral version if possible.

As for MSW-MSA attention, it seems fine with ControlNet and no special handling is required. Enable it or not according to
your preference.

## Simple Nodes

First: I strongly recommend at least skimming the **Use case** and *Compatibility note* sections of the
advanced nodes so you know when to use them and potential problems to avoid. That information won't be repeated here.
Also, I still haven't found the best  combination of settings so it is likely the preset parameters for these nodes
will change in the future.

When the nodes activate, they will output some information to your log like this:

```plaintext
** ApplyRAUNetSimple: Using preset SD15 high:
  upscale bicubic,
  in/out blocks [3 / 8],
  start/end percent 0.0/0.5  |
  CA upscale bicubic,
  CA in/out blocks [1 / 11],
  CA start/end percent 0.0/0.35

** ApplyMSWMSAAttentionSimple: Using preset SD15:
  in/mid/out blocks [1,2 /  / 11,10,9],
  start/end percent 0.2/1.0
```

(Example split into multiple lines for readability.)

If you want reproducible generations, take note of those settings: you can enter them into the advanced nodes.

### `ApplyMSWMSAAttentionSimple`

Simplified version of the MSW-MSA attention node. Use the `SD15` setting for SD 2.1 as well.

### `ApplyRAUNetSimple`

Simplified version of the `ApplyRAUNet` node. All the same caveats apply. Use the `SD15` for SD 2.1 as well.

*Note*: This node just chooses a preset, so it's not necessarily important for your resolution to match
the `res_mode` setting.

## Advanced Nodes

Common inputs:

**Time mode**: You can set a range when the node is active. May be `percent` (1.0 is 100%) - however note that
this is based on sampling percentage completed and _not_ percentage of steps completed. You may also specify
times using `timestep`s or raw `sigma` values (I recommend using percentages normally). Start and end time
properties should be specified using the mode you choose. *Note*: Time mode only controls the format
you enter start/end times with, it doesn't change the behavior of the nodes at all. If you don't know what
timesteps or sigmas are, just use `percent`.

**Blocks**: A comma separated list of block numbers. Input blocks are also known as down blocks, output blocks
are also known as up blocks. SD1.5 and SDXL at least also have one middle block. For visualizing when blocks
are active, imagine a simple model with 3 input and output blocks and one middle block. In that case evaluating
the model would look like:

```plaintext
 start       end
   |          ^
   v          |
input 0    output 2
   |          |
input 1    output 1  <- now you know why they call it a u-net.
   |          |
input 2    output 0
   |          |
   \ middle 0 /
```

This is important because if you downscale `input 0`, you'd want to reverse the operation in the corresponding
block which would _not_ be `output 0` (it would be `output 2`).

### `ApplyMSWMSAAttention`

**Use case**: Performance improvement for SD 1.5, may improve generation quality at high res for both
SD 1.5 and SDXL.

Applies MSW-MSA attention. Note that this probably won't work with other attention modifications like
perturbed attention, self-attention guidance, nearsighted attention, etc. This is a performance boost for
SD 1.5 and it seems like it may also reduce artifacts at least at high res (subjective, not scientific
opinion). I made a small change compared to the reference implementation: I ensure that the shifts used
are different each step.

The default block values are for SD 1.5.

**SD 1.5**:

| Type | Attention Blocks |
| - | - |
| Input (down) | 1, 2, 4, 5, 7, 8 |
| Middle | 0 |
| Output (up) | 3, 4, 5, 6, 7, 8, 9, 10, 11 |

Recommended SD 1.5 settings: input `1, 2`, output `9, 10, 11`.

**SDXL**

| Type | Attention Blocks |
| - | - |
| Input (down) | 4, 5, 7, 8 |
| Middle | 0 |
| Output (up) | 0, 1, 2, 3, 4, 5 |

Recommended SDXL settings: input `4, 5`, output `4, 5`.

*Note*: This doesn't seem to help performance much with SDXL. Also at very extreme resolutions (over 2048) you
may need to set MSW-MSA attention to start a bit later. Try starting at 0.2 or after other scaling effects end.

*Compatibility note*: If you run into tensor size mismatch errors, try using images sizes that are multiples
of 32, 64 or 128 (may need to experiment). Known to work with ELLA, FreeU (V2), CFG rescaling effects, SAG
 and PAG. Likely does not work with HyperTile, Deep Cache, Nearsighted/Slothful attention or other attention
 patches that affect the same blocks (SAG/PAG normally target the middle block which is fine).

***

Input blocks downscale and output blocks upscale so the biggest effect on performance will be applying this
to input blocks with a low block number and output blocks with a high block number.

<details>

<summary>YAML parameters</summary>

This input can be converted to a multi-line text widget. Allows setting advanced/rare parameters. You can also override the node parameters here. JSON is valid YAML so you can use that if you prefer.

Default parameter values:

```yaml
# In addition to the extra advanced options, you can override any fields from
# the node here. For example:
# time_mode: percent

# Scale mode used as a fallback only when image sizes are not multiples of 64. May decrease image quality.
# May also be set to "disabled" to disable the workaround or "skip" to skip MSW-MSA attention on incompatible sizes.
scale_mode: nearest-exact

# Scale mode used to reverse the scale_mode scaling.
reverse_scale_mode: nearest-exact

# One of global, block, both, ignore
last_shift_mode: global

# One of decrement, increment, retry
last_shift_strategy: decrement

# Can be enabled to disable the log warning about incompatible image sizes.
silent: false

# Allow scaling the window before/after the window or window reverse operation.
pre_window_multiplier: 1.0
post_window_multiplier: 1.0
pre_window_reverse_multiplier: 1.0
post_window_reverse_multiplier: 1.0

# Positive/negative distance to search for candidate rescales when dealing with incompatible
# resolutions. Can possibly be used to brute force attn2 application (you can set it to something
# absurd like 32).
rescale_search_tolerance: 1

# Not recommended. Forces applying the attention patch to attn2.
force_apply_attn2: false

# Enable extra logging output. (Currently only dumps the parameters.)
verbose: false
```


* `scale_mode`: Scale mode used as a fallback only when image sizes are not multiples of 64. May decrease image quality. Use `disabled` to bypass the fallback (may result in error) or `skip` to skip using MSW-MSA attention when the image size is incompatible. Any of the available scaling modes may be used here.
* `reverse_scale_mode`: Scale mode used to reverse the scaling done by `scale_mode`. No effect when `scale_mode` is not being applied.
* `last_shift_mode`: `global` - tracking is independent of blocks. `block` - remembers the last shift by block. `both` - avoids using the last shift both by block and globally. `ignore` - just uses whatever shift was randomly picked.
* `last_shift_strategy`: Only has an effect when `last_shift_mode` is not `ignore`. There are four possible shift types. `decrement` - decrements the shift type. `increment` - increments the shift type. `retry` - keeps generating random shifts until it hits one not on the ignore list (changes seeds most significantly).
* `pre_window_multipler` (etc): You can multiply the tensor before/after the window or window reverse operation. There's generally no difference between doing it before or after unless you're using weird upscale modes from [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh). I don't know why/when this would be useful, but it's there if you want to mess with it!
* `force_apply_attn2`: Forces applying to attn2 rather than attn1. **Warning**: MSW-MSA attention was not made for `attn2` and the sizes are guaranteed to be incompatible and require scaling. Using it also doesn't seem to improve performance, there isn't much reason to enable this unless you're a weirdo like me and just like trying strange things.

The last shift options are for trying to avoid choosing the same shift size consecutively. This may or may not actually be helpful.

**Note**: Normal error checking generally doesn't apply to parameters set/overriden here. You are allowed to shoot yourself in the foot and will likely just get an exception if you enter the wrong type/an absurd value.

</details>

### `ApplyRAUNet`

**Use case**: Helps avoid artifacts when generating at resolutions significantly higher than what the model
normally supports. Not beneficial when generating at low resolutions (and actually likely harms quality).
In other words, only use it when you have to.

As above, the default block values are for SD 1.5.

CA blocks are (maybe?) cross attention. The blocks you can target are the same as the self-attention blocks
listed above.

Non-CA blocks are used to target upsampler and downsampler blocks. When setting an input block, you must use
the corresponding output block. For example, if you're using SD 1.5 and you set input 3 then you must set
output 8. This also applies when setting CA blocks. SD 1.5 has 12 blocks on each side of the middle block, SDXL has 9.

**SD 1.5**:

| Input (down) Block | Output (up) Block |
| - |  - |
| 3 | 8 |
| 6 | 5 |
| 9 | 2 |

Recommended SD 1.5 settings:

1. input 3, output 8, CA input 4, CA output 8, start 0.0, end 0.45, CA start 0.0, CA end 0.3 - I believe this
   is close to what the official implementation uses.
2. input 3, output 8, CA input **1**, CA output **11**, start 0.0, end 0.6, CA start 0.0, CA end 0.35 - Seems to
   work better than the above for me at least when generating at fairly high resolutions (~2048x2048).

Example workflow: [Image with embedded SD1.5 workflow](assets/sd15_workflow.png)

**SDXL**:

| Input Downsample block | Output Upsample Block |
| - |  - |
| 3 | 5 |
| 6 | 2 |

Recommended SDXL settings: In general I haven't seen amazing results with SDXL. You can try using
input 3, output 5 and disabling CA (set the `ca_start_time` to 1.0) _or_ setting CA input 2, CA output 7
and disabling the upsampler/downsampler patch (set `start_time` to 1.0). I don't recommend leaving both
enabled at the same time, but feel free to experiment. SDXL seems very sensitive to these settings. Also I don't
recommend enabling RAUNet at all unless you are generating at a resolution significantly higher than what the
model supports. Using an ancestral or SDE sampler seems to work best with SDXL and RAUNet.

Why does setting input 2 correspond with output 7? I actually have no idea, I would have expected it to be
6.

Example workflow: [Image with embedded SDXL workflow](assets/sdxl_workflow.png)

***

For upscale mode, good old `bicubic` may be best. The second best alternative is probably `bislerp`. Two
step upscale does does half of the upscale with nearest-exact and the remaining half with the upscale method you
selected. The difference seems very minor and I am not sure which setting is better.

If you have my [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh) nodes active, there will be more
upscale options. The `random` upscale method seems to work pretty well, possibly also `random+renoise1` (
  adds a small amount of gaussian noise after upscaling
).

This node works well with [restart sampling](https://github.com/ssitu/ComfyUI_restart_sampling) — you may need
to manually adjust the restart segments. Generally you don't want to restart back into the scaling effect,
rather right after it ends to give the model a chance to clean up artifacts. Using the `a1111` preset will
probably work best if you don't want to manually set segments.

*Compatibility note*: Should be compatibile with the same effects as MSW-MSA attention. Likely won't work with
other scaling effects that target the same blocks (i.e. Deep Shrink). By itself, I think it should be fine with
HyperTile and Deep Cache though I haven't actually tested that. May not work properly with ControlNet.

<details>

<summary>YAML parameters</summary>

This input can be converted to a multi-line text widget. Allows setting advanced/rare parameters. You can also override the node parameters here. JSON is valid YAML so you can use that if you prefer.

Default parameter values:

```yaml
# In addition to the extra advanced options, you can override any fields from
# the node here. For example:
# time_mode: percent

# Patches input blocks after the skip connection when enabled (similar to Kohya deep shrink).
ca_input_after_skip_mode: false

# Either null or set to a time (with the same time mode as the other times).
# Starts fading out the CA scaling effect, starting from the specified time.
ca_fadeout_start_time: null

# Maximum fadeout, specified as a percentage of the total scaling effect.
ca_fadeout_cap: 0.0

# null or float. Allows setting the width scale separately. When null the same
# factor will be used for height and width.
ca_downscale_factor_w: null

# When applying CA scaling, ensures the rescaled latent is divisible by the specified incremenrt.
ca_latent_pixel_increment: 8

# When using the avg_pool2d method, enable ceil mode.
# See: https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html
ca_avg_pool2d_ceil_mode: true

# Allows applying a multiplier to the tensor: can be set separately for before/after upscale, downscale
# and whether it's CA or not.
pre_upscale_multiplier: 1.0
post_upscale_multiplier: 1.0
pre_downscale_multiplier: 1.0
post_downscale_multiplier: 1.0
ca_pre_upscale_multiplier: 1.0
ca_post_upscale_multiplier: 1.0
ca_pre_downscale_multiplier: 1.0
ca_post_downscale_multiplier: 1.0

# Enable extra logging output. (Currently only dumps the parameters.)
verbose: false
```

* `ca_input_after_skip_mode`: When applying CA scaling, the effect will occur after the skip connection. This is the default for Kohya Deep Shrink and may produce less noisy results. **Note**: This changes the corresponding output block you need to set if not targeting a downscale block (i.e. ones you can target with the main RAUNet effect). It seems like you generally just subtract one. Example: Using SD15 and targeting input 4, you'd normally use output 8 - use output 7 instead.
* `ca_latent_pixel_increment`: Ensures the scaled sizes are a multiple of the latent pixel increment. The default of 8 should ensure the scaled size is compatible with MSW-MSA attention without scaling workarounds. *Note*: Has no effect when downscaling with `avg_pool2d`.
* `ca_fadeout_start_time`: Will start fading out the CA downscale factor starting from the specified time (which uses the same time mode as other configured times). The fadeout occurs such that the downscale factor will reach `1.0` (no downscaling) at `ca_end_time`. This can (sometimes) help decrease artifacts compared to simply ending the scale effect abruptly.
* `ca_fadeout_cap`: Only has an effect when fadeout is in effect (see above). This is expressed as a percentage of the scaling effect, so, for example, you could set it to `0.5` to fade out the first 50% of the downscale effect and after that the downscale would stay at 50% (of the total downscale effect) until `ca_end_time` is reached.
* `pre_upscale_multipler` (etc): You can multiply the tensor before/after it's upscaled or downscaled. There's generally no difference between doing it before or after unless you're using weird upscale modes from [ComfyUI-bleh](https://github.com/blepping/ComfyUI-bleh). Should you multiply it? Maybe not! It's a setting to possibly mess with and (not very scientifically) it seems like applying a mild positive multiplier can help.

**Note**: Normal error checking generally doesn't apply to parameters set/overriden here. You are allowed to shoot yourself in the foot and will likely just get an exception if you enter the wrong type/an absurd value.

</details>

## Credits

Code based on the HiDiffusion original implementation: https://github.com/megvii-research/HiDiffusion

RAUNet backend refactored by [pamparamm](https://github.com/pamparamm) to avoid the need for monkey patching ComfyUI's Upsample/Downsample blocks.

Thanks!
