# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates registration for image-based registration."""

import logging
from pathlib import Path
from typing import Optional

from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from pydantic import validate_call

from abbott.registration.fractal_helper_tasks import (
    histogram_matching,
    pad_to_max_shape,
)

logger = logging.getLogger(__name__)


@validate_call
def compute_registration_warpfield(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    level: int,
    wavelength_id: str,
    histogram_normalisation: bool = True,
    path_to_registration_recipe: Optional[str] = None,
    save_reg_video: bool = False,
    roi_table: str = "FOV_ROI_table",
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Calculate warpfield registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the registration folder of the
       moving acquisition.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be used for registration.
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        histogram_normalisation: If `True`, applies histogram normalisation to the
            moving image before calculating the registration. Default: `True`.
        path_to_registration_recipe: Path to the warpfield .yml registration recipe.
            This parameter is optional, if not provided, the default .yml recipe
            will be used.
        save_reg_video: If `True`, saves the video showing the registration in
            registration folder. Default: `False`.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        use_masks:  If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `embryo_ROI_table`).
        masking_label_name: Optional label for masking ROI e.g. `embryo`.
        overwrite: If `True`, overwrite existing registration files.
            Default: `False`.
    """
    try:
        import cupy as cp
        import warpfield
    except ImportError as e:
        raise ImportError(
            "The `compute_registration_warpfield` task requires GPU. "
        ) from e

    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating warpfield registration per {roi_table=} for "
        f"{wavelength_id=}."
    )

    reference_zarr_url = init_args.reference_zarr_url

    # Load channel to register by
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    channel_index_ref = ome_zarr_ref.get_channel_idx(wavelength_id=wavelength_id)

    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    channel_index_align = ome_zarr_mov.get_channel_idx(wavelength_id=wavelength_id)

    # Get images for the given level and at highest resolution
    ref_images = ome_zarr_ref.get_image(path=str(level))
    mov_images = ome_zarr_mov.get_image(path=str(level))

    # Read ROIs
    if use_masks:
        ref_roi_table = ome_zarr_ref.get_masking_roi_table(roi_table)
        mov_roi_table = ome_zarr_mov.get_masking_roi_table(roi_table)
    else:
        ref_roi_table = ome_zarr_ref.get_table(roi_table)
        mov_roi_table = ome_zarr_mov.get_table(roi_table)

    # Masked loading checks
    if use_masks:
        if ref_roi_table.table_type() != "masking_roi_table":
            logger.warning(
                f"ROI table {roi_table} in reference OME-Zarr is not "
                "a masking ROI table. Falling back to use_masks=False."
            )
            use_masks = False
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False

        ref_images = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path=str(level),
        )

        mov_images = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            path=str(level),
        )

    logger.info(
        f"Found {len(ref_roi_table.rois())} ROIs in {roi_table=} to be processed."
    )

    # For each acquisition, get the relevant info
    # TODO: Add additional checks on ROIs?
    if len(ref_roi_table.rois()) != len(mov_roi_table.rois()):
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "acquisitions (e.g. well, FOV ROIs). Here, the ROIs in the "
            f"reference acquisitions were {len(ref_roi_table.rois())}, but the "
            f"ROIs in the alignment acquisition were {mov_roi_table.rois()}."
        )

    # Read full-res pixel sizes from zarr attributes
    pxl_sizes_zyx_ref = ref_images.pixel_size.zyx
    pxl_sizes_zyx_mov = mov_images.pixel_size.zyx

    if pxl_sizes_zyx_ref != pxl_sizes_zyx_mov:
        raise ValueError(
            "Pixel sizes need to be equal between acquisitions "
            "for warpfield registration."
        )

    # Load warpfield recipe
    if path_to_registration_recipe is not None:
        try:
            recipe = warpfield.Recipe.from_yaml(path_to_registration_recipe)
        except Exception as e:
            raise ValueError(
                "Failed to load registration recipe from "
                f"{path_to_registration_recipe}. "
                "Please check the file path and format."
            ) from e
    else:
        recipe = warpfield.Recipe.from_yaml("default.yml")

    logger.info(
        f"Start of warpfield registration for {zarr_url=} with registration {recipe=}"
    )

    num_ROIs = len(ref_roi_table.rois())
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        ROI_id = ref_roi.name
        logger.info(f"Now processing ROI {i_ROI + 1}/{num_ROIs} for {wavelength_id=}.")

        if use_masks:
            img_ref = ref_images.get_roi_masked(
                label=int(ROI_id),
                c=channel_index_ref,
            ).squeeze()
            img_mov = mov_images.get_roi_masked(
                label=int(ROI_id),
                c=channel_index_align,
            ).squeeze()

        else:
            img_ref = ref_images.get_roi(
                roi=ref_roi,
                c=channel_index_ref,
            ).squeeze()
            mov_roi = [roi for roi in mov_roi_table.rois() if roi.name == ref_roi.name][
                0
            ]
            img_mov = mov_images.get_roi(
                roi=mov_roi,
                c=channel_index_align,
            ).squeeze()

        # Pad images to the same shape
        # Calculate maximum dimensions needed
        max_shape = tuple(
            max(r, m) for r, m in zip(img_ref.shape, img_mov.shape, strict=False)
        )
        img_ref = pad_to_max_shape(img_ref, max_shape)
        img_mov = pad_to_max_shape(img_mov, max_shape)

        # Apply histogram normalisation if requested
        if histogram_normalisation:
            img_mov = histogram_matching(img_mov, img_ref)
            logger.info(
                f"Applied histogram normalisation to moving image for ROI {ROI_id}."
            )

        ##############
        #  Calculate the transformation
        ##############
        # Adjust block size so that it fits the ROI shape if necessary
        recipe_adjusted = recipe.model_copy()
        for i, reg_level in enumerate(recipe_adjusted.levels):
            block_sizes = reg_level.block_size
            original_blocksizes = block_sizes.copy()

            for dim, (img_shape, block_size) in enumerate(
                zip(list(img_mov.shape), block_sizes)
            ):
                if img_shape < block_size:  # adjust if necessary
                    block_sizes[dim] = img_shape

            # Only apply change if blocksize was modified
            if block_sizes != original_blocksizes:
                logger.warning(
                    f"Blocksize {original_blocksizes} too large for ROI "
                    f"of shape {img_mov.shape}. "
                    f"Decreased blocksize of level {i} to {block_sizes}."
                )
                recipe_adjusted.levels[i].block_size = block_sizes

        # Start registration
        pixel_sizes_zyx = ome_zarr_ref.get_image(path=str(level)).pixel_size.zyx
        callback = warpfield.utils.mips_callback(units_per_voxel=pixel_sizes_zyx)

        # Check if registration video should be saved
        video_path = None
        if save_reg_video:
            video_path = (
                Path(zarr_url) / "registration" / f"registration_roi_{ROI_id}.mp4"
            )
            video_path.parent.mkdir(exist_ok=True, parents=True)
            video_path = video_path.as_posix()

        _, warp_map, _ = warpfield.register_volumes(
            img_ref,
            img_mov,
            recipe_adjusted,
            callback=callback,
            video_path=video_path,
        )

        # Write transform parameter files
        fn = (
            Path(zarr_url)
            / "registration"
            / (f"{roi_table}_roi_{ROI_id}_lvl_{level}.h5")
        )
        if fn.exists() and not overwrite:
            raise FileExistsError(
                f"Registration file {fn} already exists. To overwrite, "
                "set the `overwrite` parameter to True."
            )
        fn.parent.mkdir(exist_ok=True, parents=True)
        warp_map.to_h5(fn)

        # Free GPU memory after each ROI
        cp.get_default_memory_pool().free_all_blocks()

        logger.info(
            "Finished computing warpfield registration parameters "
            f"for ROI {ROI_id}, saving params to {fn}."
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_registration_warpfield,
        logger_name=logger.name,
    )
