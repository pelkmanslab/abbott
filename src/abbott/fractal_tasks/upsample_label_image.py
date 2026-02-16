"""Fractal task to upsample a label image to highest image resolution."""

import logging
from typing import Optional

import fractal_tasks_core
from fractal_tasks_core.utils import _split_well_path_image_path
from ngio import open_ome_zarr_container
from ngio.io_pipes import dask_match_shape
from ngio.utils._errors import NgioValueError
from pydantic import validate_call

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_call
def upsample_label_image(
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    label_name: str,
    ref_acquisition: Optional[int] = None,
    output_label_name: Optional[str] = None,
    output_ROI_table: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """Upsample a label image to match level 0 image resolution.

    This task loads the label image, upsamples it to the highest resolution,
    consolidates the label image to all other levels and saves it back to the
    OME-Zarr file.

    This task is useful if e.g. segmentation was performed at a lower resolution
    due to memory constraints.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label image to upsample.
        ref_acquisition: Optional, if provided the task will not cause an error
            if the label does not exist for non-reference acquisitions.
        output_label_name: Optionally new label name for the upsampled label image.
        output_ROI_table: If provided, a  masking ROI table with that name is created,
            which will contain the bounding boxes of the newly upsampled
            labels. ROI tables should have `ROI` in their name.
        overwrite: If `True`, overwrite existing label and ROI table (if set).
    """
    logger.info(f"Starting label upsampling for {zarr_url=} and {label_name=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    if output_label_name is None:
        output_label_name = label_name

    # Read attributes from lvl 0 image
    image_lvl_0 = ome_zarr.get_image(path="0")
    full_res_shape = image_lvl_0.shape[-3:]  # ignore channel dimension

    # If ref_acquisition is provided, check if label exists for that
    # acquisition and don't throw error if it doesn't exist
    if ref_acquisition is not None:
        well_path, img_path = _split_well_path_image_path(zarr_url)
        ref_zarr_url = f"{well_path}/{ref_acquisition}"
        if zarr_url != ref_zarr_url:
            try:
                ome_zarr.get_label(label_name, path="0")
            except NgioValueError:
                logger.warning(
                    f"Label {label_name} not found for acquisition "
                    f"{img_path}. Skipping upsampling."
                )
                return None

    # Read attributes from lower res label image
    label_low_res = ome_zarr.get_label(label_name, path="0")
    actual_res_shape = label_low_res.shape

    # Check if upsampling is needed
    if full_res_shape == actual_res_shape:
        return ValueError(
            "Label image already at full resolution. " "No upsampling needed."
        )

    ##############
    #  Start upsampling
    ##############

    # Get low res label image as dask array
    label_low_res_da = label_low_res.get_array(mode="dask")

    # Upsample label image
    logger.info(f"Upsampling label from shape {actual_res_shape} to {full_res_shape}")

    upsampled_label_da = dask_match_shape(
        label_low_res_da,
        full_res_shape,
        array_axes=label_low_res.axes,
        reference_axes=image_lvl_0.axes[-3:],
    )

    # Ensures chunk shapes are computed correctly
    upsampled_label_da.compute_chunk_sizes()
    logger.info(f"Upsampled label shape: {upsampled_label_da.shape}")

    # Write upsampled label image to disk
    mask = ome_zarr.derive_label(
        output_label_name,
        axes_names=label_low_res.axes,
        shape=upsampled_label_da.shape,
        chunks=upsampled_label_da.chunksize,
        overwrite=overwrite,
    )

    mask.set_array(upsampled_label_da)
    mask.consolidate()

    # Build ROI table if requested
    if output_ROI_table is not None:
        masking_table = ome_zarr.build_masking_roi_table(output_label_name)
        ome_zarr.add_table(output_ROI_table, masking_table, overwrite=overwrite)

    logger.info(f"Successfully upsampled {label_name} to {output_label_name=}")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=upsample_label_image,
        logger_name=logger.name,
    )
