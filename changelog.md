# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20241224

Reworked approach to integrating with external node packs. This _shouldn't_ cause any visible changes from a user perspective but please create an issue if you notice anything weird.

## 20241014

_Note_: Advanced MSW-MSA Attention node parameters changed. May break workflows.

_Note_: This update may slightly change seeds.

* MSW-MSA attention can now work with all images sizes. When the size is incompatible it will scale the latent which may affect quality. Contributed by @pamparamm. Thanks!
* Scaling now tries to make the output size a multiple of 8 so it's compatible with MSW-MSA attention. May change seeds, set `ca_latent_pixel_increment: 1` in YAML parameters for the old behavior. *Note*: Does not apply if you use `avg_pool2d` for downscaling.
* CA downscaling now uses `adaptive_avg_pool2d` as the default method which supports fractional downscale sizes. As far as I know, it's the same as `avg_pool2d` with integer sizes but it's possible this will change seeds.
* Simple nodes now support an "auto" model type parameter that will try to guess the model from the latent type.
* Added a `yaml_parameters` input to the advanced nodes which allows specifying advanced/uncommon parameters. See main README for possible settings.
* You can now use a different scale factor for width and height in RAUNet CA scaling. See `ca_downscale_factor_w` in YAML parameters.
* You can now fade out the CA scaling effect in RAUNet node. See `ca_fadeout_start_time` and `ca_fadeout_cap` in YAML parameters.
* Simple nodes default parameters for SDXL models adjusted to match the official HiDiffusion ones more closely.

Check the expandable "YAML Parameters" sections in the main README for more information about advanced parameters added in this update.

## 20240827

* Fixed (hopefully) an issue with RAUNet model patching that could cause semi-non-deterministic output. Unfortunately the fix also may change seeds.

## 20240813

_Note_: Advanced RAUNet node parameters changed, will break workflows.

_Note_: May (slightly) change seeds.

* RAUNet: Patch backend refactored to work as a normal model patch.
* RAUNet: You can now set the CA downscale factor in the advanced node (basically works like deep shrink).
* RAUNet: You can now set the CA downscale mode in the advanced node (previously always was `avg_pool2d`).

## 20240802

_Note_: May change seeds, for previous behavior set two stage upscale mode to `nearest-exact` in the advanced RAUNet node.

_Note_: Advanced RAUNet node parameters changed, will break workflows.

* RAUNet: Fixed issue where two stage upscale setting didn't work properly.
* RAUNet: Two stage upscale is no longer a toggle and allows choosing the scale method.
