### Purpose
Computes **image-based channel registration** transformations for OME-Zarr images using the [elastix](https://elastix.lumc.nl/) library.

For each ROI, all channels (except the reference) are summed and registered against the reference channel using e.g. a **SimilarityTransform** (or any elastix transform configured via parameter files). The resulting transformation parameters are stored in a `GenericTable` within the OME-Zarr image.

Registration can be performed on intensity or label images (in case intensity images to be registered contain little similarity).

Typically used as the **first task** in a channel registration workflow, followed by `Apply Channel Registration (elastix)`.

### Workflow context
- **This task** — computes per-ROI transformations and saves them to a table.
- **`Apply Channel Registration (elastix)`** — reads the transformation table written by this task and applies it to the images.

### Limitations
- Masking (`use_masks=True`) requires a masking ROI table; falls back to unmasked loading with a warning if not available.
