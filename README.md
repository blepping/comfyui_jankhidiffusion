# ComfyUI jank HiDiffusion

Janky experimental attempt at implementing [HiDiffusion](https://github.com/megvii-research/HiDiffusion) for ComfyUI.

## Description

Read the link above for an official description. The following is just my understanding and may or may not be
correct:

As far as I understand it, the RAU-Net part is essentially Kohya Deep Shrink (AKA `PatchModelAddDownscale`) with a
different name. The main difference for that part is the downscale methods - it uses convolution with stride/dilation
and pool averaging to downscale while Deep Shrink usually uses bicubic downscaling.

Not sure how to describe MSW-MSA attention. It seems like a big performance boost for SD 1.5 at high res and also
appears to increase quality. Note that it does not enable high res generation by itself.

## Caveats

I'm not an expert on diffusion stuff and I'm not sure I fully understood the HiDiffusion code. My implementation
may or may not work correctly. If you experience issues, please don't blame HiDiffusion unless you can also reproduce
it with their implementation.

**Important**: The node default values are for SD 1.5, they won't work well for other models (like SDXL). This
stuff probably doesn't work at all for more exotic models like Cascade.

* Not all aspect ratios work with the MSW-MSA attention node. It may be the same with the original implementation?
* The MSW-MSA attention node doesn't seem to help performance with SDXL much.
* ComfyUI doesn't have built-in support for patching the Upscale/Downscale model blocks and in fact doesn't pass
  information like the timestep to them. I had to monkeypatch some of ComfyUI's guts. This means if Comfy changes
  something, my code will likely break horribly.
* Some other custom nodes will also try to patch ComfyUI's internals - notably FreeU Advanced. I included a workaround
  that will at least let both custom nodes be loaded at the same time. It may or may not work with the actual FreeU
  Advanced node. Also there may be other custom nodes/collections that will cause issues.
* I may not have implemented the cross-attention block part correctly. As far as I could tell, it seemed like it
  was just patching a normal block, not actual cross-attention (so almost exactly like Deep Shrink). My version
  is implemented that way and does not patch cross-attention.
* Customizable, but not very userfriendly. You get to figure out the blocks to patch!
* Since ComfyUI doesn't allow applying a model patch for some of the RAUNet stuff, the patched Upscale/Downscale blocks
  are _always_ active and will delegate to the original versions when RAUNet is disabled. This means having these nodes
  loaded can break stuff even if you're not actually using them!
* The list of caveats is too long, and it's probably not even complete. Yikes!

## Nodes

Common inputs:

**Time mode**: You can set a range when the node is active. May be `percent` (1.0 is 100%) - however note that
this is based on sampling percentage completed and _not_ percentage of steps completed. You may also specify
times using `timestep`s or raw `sigma` values (I recommend using percentages normally). Start and end time
properties should be specified using the mode you choose.

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

Applies MSW-MSA attention. Note that this probably won't work with other attention modifications like
perturbed attention, self-attention guidance, nearsighted attention, etc. This is a performance boost for
SD 1.5 and it seems like it may also reduce artifacts at least at high res (subjective, not scientific
opinion). I made a small change compared to the reference implementation: I ensure that the shifts used
are different each step.

The default block values are for SD 1.5. From what I can see, these are possible self-attention blocks:

**SD 1.5**:

Input: 1,2,4,5,7,8

Output: 11,10,9,8,7,6,5,4,3

**SDXL**

Input: 4,5,7,8

Output: 5,4,3,2,1,0

***

Input blocks downscale and output blocks upscale so the biggest effect on performance will be applying this
to input blocks with a low block number and output blocks with a high block number.

For SDXL you can try using input `4,6`, output `5,4`.

### `ApplyRAUNet`

First, an important note: half of the RAUNet implementation is not a normal model patch. It globally patches
ComfyUI. This means if you actually want to disable it after it's been enabled, you **must** execute the
workflow once with the node toggled to disabled (at let execution reach that point). **Note**: If you don't
do that, the RAUNet changes will be active even with the node disabled, muted or deleted entirely.

Also note that you should only use one `ApplyRAUNet` node.

As above, the default block values are for SD 1.5.

CA blocks are (maybe?) cross attention. The blocks you can target are the same as the self-attention blocks
listed above.

Non-CA blocks are upsampler and downsampler blocks. I believe these are the possible values:

**SD 1.5**:

Input (downsampling): 3,6,9
Output (upsampling): 8,5,2

**SDXL**:

Input (downsampling): 3,6
Output (upsampling): 5,2

Remember to pair the blocks, i.e. if you're using SDXL and set input 3 you need to set output 8. If you set
CA input 1, you need to set CA output 11. For SD 1.5 I believe there are 12 input blocks, SDXL 9 (maybe?).
The corresponding output block should be something like `num_input_blocks - input_block`. For SDXL you can try
using input `3`, output `5`, CA input `5`, CA output `4`. The CA part doesn't seem to help much with SDXL,
you can simply disable it by setting `ca_start_time` to `1.0`.

For upscale mode, good old `bicubic` may be best. The second best alternative is probably `bislerp`.

## Credits

Code based on the HiDiffusion original implementation: https://github.com/megvii-research/HiDiffusion

Thanks!
