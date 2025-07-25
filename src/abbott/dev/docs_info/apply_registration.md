### Purpose
- **Applies pre-calculated registration** from `Compute Registration (elastix)` task to images in an **HCS** OME-Zarr dataset, aligning all acquisitions to a specified reference acquisition.
- Replaces the non-aligned image with the newly aligned image in the dataset if `overwrite input` is selected.
- Typically used as the second task in a workflow, following `Compute Registration (elastix)`.

### Limitations
- If `overwrite input` is selected, the non-aligned image is permanently deleted, which may impact workflows requiring access to the original images.
