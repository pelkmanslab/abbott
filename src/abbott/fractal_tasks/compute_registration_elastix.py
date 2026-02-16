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

import numpy as np
import pandas as pd
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from ngio import open_ome_zarr_container
from ngio.tables import GenericTable
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from abbott.registration.conversions import to_itk
from abbott.registration.fractal_helper_tasks import pad_to_max_shape
from abbott.registration.itk_elastix import (
    create_identity_transform_from_file,
    get_identity_parameter_file_path,
    register_transform_only,
)

logger = logging.getLogger(__name__)


@validate_call
def compute_registration_elastix(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    level: int = 0,
    ref_wavelength_id: str,
    mov_wavelength_id: Optional[str] = None,
    parameter_files: list[str],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
    skip_failed_rois: bool = True,
) -> None:
    """Calculate elastix registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating the transformation for that ROI
    3. Storing the calculated transformation in the ROI table

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution. Currently only level 0
            is supported.
        ref_wavelength_id: Wavelength that will be used for image-based
            registration as the reference; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        mov_wavelength_id: (Optional) wavelength that will be used for image-based
            registration for moving images; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        parameter_files: Paths to the elastix parameter files to be used. List order is
             order of registration. E.g. parse first rigid, then affine
             and lastly bspline.
        lower_rescale_quantile: Lower quantile for rescaling the image
            intensities before applying registration. Can be helpful
             to deal with image artifacts. Default is 0.
        upper_rescale_quantile: Upper quantile for rescaling the image
            intensities before applying registration. Can be helpful
            to deal with image artifacts. Default is 0.99.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        use_masks:  If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `embryo_ROI_table`).
        masking_label_name: Optional label for masking ROI e.g. `embryo`.
        skip_failed_rois: If `True`, ROIs that fail during registration
            will be skipped and the task will continue with the next ROI. An
            identity transformation will be written for the failed ROIs and a
            condition table will be added to the OME-Zarr listing the ROIs
            with issues.

    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating elastix registration per {roi_table=} for "
        f"{ref_wavelength_id=}."
    )

    reference_zarr_url = init_args.reference_zarr_url

    # Load channel to register by
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    channel_index_ref = ome_zarr_ref.get_channel_idx(wavelength_id=ref_wavelength_id)

    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    if mov_wavelength_id is not None:
        channel_index_align = ome_zarr_mov.get_channel_idx(
            wavelength_id=mov_wavelength_id
        )
        logger.info(f"Running registration with {mov_wavelength_id=}")
    else:
        channel_index_align = ome_zarr_mov.get_channel_idx(
            wavelength_id=ref_wavelength_id
        )

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

    # Read pixel sizes from zarr attributes
    pxl_sizes_zyx_ref_full_res = ome_zarr_ref.get_image(path="0").pixel_size.zyx
    pxl_sizes_zyx_mov_full_res = ome_zarr_mov.get_image(path="0").pixel_size.zyx
    pxl_sizes_zyx_ref = ome_zarr_ref.get_image(path=str(level)).pixel_size.zyx
    pxl_sizes_zyx_mov = ome_zarr_mov.get_image(path=str(level)).pixel_size.zyx

    if pxl_sizes_zyx_ref_full_res != pxl_sizes_zyx_mov_full_res:
        raise ValueError(
            "Pixel sizes need to be equal between acquisitions "
            "for elastix registration."
        )

    num_ROIs = len(ref_roi_table.rois())
    failed_registrations = []
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        ROI_id = ref_roi.name
        logger.info(
            f"Now processing ROI {ROI_id} ({i_ROI + 1}/{num_ROIs}) "
            f"for {ref_wavelength_id=}."
        )

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

        # Rescale the images
        img_ref = rescale_intensity(
            img_ref,
            in_range=(
                np.quantile(img_ref, lower_rescale_quantile),
                np.quantile(img_ref, upper_rescale_quantile),
            ),
        )
        img_mov = rescale_intensity(
            img_mov,
            in_range=(
                np.quantile(img_mov, lower_rescale_quantile),
                np.quantile(img_mov, upper_rescale_quantile),
            ),
        )

        ##############
        #  Calculate the transformation
        ##############
        if img_ref.shape != img_mov.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between acquisitions."
            )
        ref = to_itk(img_ref, scale=pxl_sizes_zyx_ref)
        move = to_itk(img_mov, scale=pxl_sizes_zyx_mov)
        try:
            trans = register_transform_only(ref, move, parameter_files)
        except RuntimeError as e:
            if skip_failed_rois:
                logger.warning(
                    f"Registration failed for ROI {ROI_id} with the "
                    f"following error: {e}"
                )
                identity_file = get_identity_parameter_file_path(
                    ref.GetImageDimension()
                )
                trans = create_identity_transform_from_file(ref, identity_file)
                failed_registrations.append((ROI_id, "Elastix RuntimeError"))
            else:
                raise RuntimeError(
                    f"Computing registration failed for ROI {ROI_id} with the "
                    f"following error: {e}. Returning an identity transform "
                    "instead."
                ) from e

        # Write transform parameter files
        # TODO: Add overwrite check (it overwrites by default)
        # FIXME: Figure out where to put files
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = trans.GetParameterMap(i)
            # FIXME: Switch from ROI index to ROI names?
            fn = (
                Path(zarr_url) / "registration" / (f"{roi_table}_roi_{ROI_id}_t{i}.txt")
            )
            fn.parent.mkdir(exist_ok=True, parents=True)
            trans.WriteParameterFile(trans_map, fn.as_posix())

    # Write condition tables if registration failed for a ROI
    if failed_registrations:
        df_errors = pd.DataFrame(failed_registrations, columns=["ROI", "Reason"])
        # TODO: Update this to a condition table once we adopt ngio >= 0.3
        error_table = GenericTable(df_errors)
        # TODO: Adopt the non-experimental csv backend once we adopt ngio >= 0.3
        ome_zarr_mov.add_table(
            name="registration_errors",
            table=error_table,
            overwrite=True,
            backend="experimental_csv_v1",
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_registration_elastix,
        logger_name=logger.name,
    )
