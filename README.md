# ComfyUI jank HiDiffusion

Janky experimental implementation of [HiDiffusion](https://github.com/megvii-research/HiDiffusion) for
[ComfyUI](https://github.com/comfyanonymous/ComfyUI).

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
* ComfyUI doesn't have built-in support for patching the Upscale/Downscale model blocks and in fact doesn't pass
  information like the timestep to them. I had to monkeypatch some of ComfyUI's guts. This means if Comfy changes
  something in that area, my code will likely break horribly.
* Some other custom nodes will also try to patch ComfyUI's internals - notably FreeU Advanced. I included a workaround
  that will at least let both custom nodes be loaded at the same time. It may or may not work with the actual FreeU
  Advanced node. Also there may be other custom nodes/collections that will cause issues.
* I may not have implemented the cross-attention block part correctly. As far as I could tell, it seemed like it
  was just patching a normal block, not actual cross-attention (so almost exactly like Deep Shrink). My version
  is implemented that way and does not patch cross-attention.
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

If you find the workaround to cause issues, set the environment variable `JANKHIDIFFUSION_DISABLE_CONTROLNET_WORKAROUND`
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

### `ApplyRAUNet`

**Use case**: Helps avoid artifacts when generating at resolutions significantly higher than what the model
normally supports. Not beneficial when generating at low resolutions (and actually likely harms quality).
In other words, only use it when you have to.

You should only use one `ApplyRAUNet` node.

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
model supports.

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

## Credits

Code based on the HiDiffusion original implementation: https://github.com/megvii-research/HiDiffusion

Thanks!
