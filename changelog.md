# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

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
