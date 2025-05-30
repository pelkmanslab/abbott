# Based on https://github.com/MaksHess/abbott
"""Calculates translation for image-based registration."""

import logging
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.channels import OmeroChannel, get_channel_from_image_zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from fractal_tasks_core.tasks.io_models import InitArgsRegistration
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from abbott.io.conversions import to_itk
from abbott.registration.itk_elastix import register_transform_only

logger = logging.getLogger(__name__)


@validate_call
def compute_registration_elastix(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    wavelength_id: str,
    parameter_files: list[str],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",  # TODO: allow "emb_ROI_table"
    level: int = 2,
) -> None:
    """Calculate registration based on images

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
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        parameter_files: Paths to the elastix parameter files to be used.
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
        level: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution.

    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating translation registration per {roi_table=} for "
        f"{wavelength_id=}."
    )

    init_args.reference_zarr_url = init_args.reference_zarr_url

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(init_args.reference_zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Get channel_index via wavelength_id.
    # Intially only allow registration of the same wavelength
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=init_args.reference_zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_ref = channel_ref.index

    channel_align: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=wavelength_id,
    )
    channel_index_align = channel_align.index

    # Lazily load zarr array
    data_reference_zyx = da.from_zarr(f"{init_args.reference_zarr_url}/{level}")[
        channel_index_ref
    ]
    data_alignment_zyx = da.from_zarr(f"{zarr_url}/{level}")[channel_index_align]

    # Read ROIs
    ROI_table_ref = ad.read_zarr(f"{init_args.reference_zarr_url}/tables/{roi_table}")
    ROI_table_x = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    logger.info(f"Found {len(ROI_table_x)} ROIs in {roi_table=} to be processed.")

    # Check that table type of ROI_table_ref is valid. Note that
    # "ngff:region_table" and None are accepted for backwards compatibility
    valid_table_types = [
        "roi_table",
        "masking_roi_table",
        "ngff:region_table",
        None,
    ]
    ROI_table_ref_group = zarr.open_group(
        f"{init_args.reference_zarr_url}/tables/{roi_table}",
        mode="r",
    )
    ref_table_attrs = ROI_table_ref_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(
            f"Table '{roi_table}' (with type '{ref_table_type}') is "
            "not a valid ROI table."
        )

    # For each acquisition, get the relevant info
    # TODO: Add additional checks on ROIs?
    if (ROI_table_ref.obs.index != ROI_table_x.obs.index).all():
        raise ValueError(
            "Registration is only implemented for ROIs that match between the "
            "acquisitions (e.g. well, FOV ROIs). Here, the ROIs in the "
            f"reference acquisitions were {ROI_table_ref.obs.index}, but the "
            f"ROIs in the alignment acquisition were {ROI_table_x.obs.index}"
        )
    # TODO: Make this less restrictive? i.e. could we also run it if different
    # acquisitions have different FOVs? But then how do we know which FOVs to
    # match?
    # If we relax this, downstream assumptions on matching based on order
    # in the list will break.

    # Read pixel sizes from zarr attributes
    ngff_image_meta_acq_x = load_NgffImageMeta(zarr_url)
    pxl_sizes_zyx_full_res = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx_acq_x_full_res = ngff_image_meta_acq_x.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    pxl_sizes_zyx_acq_x = ngff_image_meta_acq_x.get_pixel_sizes_zyx(level=level)

    if pxl_sizes_zyx_full_res != pxl_sizes_zyx_acq_x_full_res:
        raise ValueError(
            "Pixel sizes need to be equal between acquisitions for " "registration."
        )

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices_ref = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )
    check_valid_ROI_indices(list_indices_ref, roi_table)

    list_indices_acq_x = convert_ROI_table_to_indices(
        ROI_table_x,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_acq_x_full_res,
    )
    check_valid_ROI_indices(list_indices_acq_x, roi_table)

    num_ROIs = len(list_indices_ref)
    compute = True
    for i_ROI in range(num_ROIs):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} " f"for channel {channel_align}."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices_ref[i_ROI]),
            compute=compute,
        )
        img_acq_x = load_region(
            data_zyx=data_alignment_zyx,
            region=convert_indices_to_regions(list_indices_acq_x[i_ROI]),
            compute=compute,
        )

        # Rescale the images
        img_ref = rescale_intensity(
            img_ref,
            in_range=(
                np.quantile(img_ref, lower_rescale_quantile),
                np.quantile(img_ref, upper_rescale_quantile),
            ),
        )
        img_acq_x = rescale_intensity(
            img_acq_x,
            in_range=(
                np.quantile(img_acq_x, lower_rescale_quantile),
                np.quantile(img_acq_x, upper_rescale_quantile),
            ),
        )

        ##############
        #  Calculate the transformation
        ##############
        if img_ref.shape != img_acq_x.shape:
            raise NotImplementedError(
                "This registration is not implemented for ROIs with "
                "different shapes between acquisitions."
            )

        ref = to_itk(img_ref, scale=tuple(pxl_sizes_zyx))
        move = to_itk(img_acq_x, scale=tuple(pxl_sizes_zyx_acq_x))
        trans = register_transform_only(ref, move, parameter_files)

        # Write transform parameter files
        # TODO: Add overwrite check (it overwrites by default)
        # FIXME: Figure out where to put files
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = trans.GetParameterMap(i)
            # FIXME: Switch from ROI index to ROI names?
            fn = Path(zarr_url) / "registration" / (f"{roi_table}_roi_{i_ROI}_t{i}.txt")
            fn.parent.mkdir(exist_ok=True, parents=True)
            trans.WriteParameterFile(trans_map, fn.as_posix())


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=compute_registration_elastix,
        logger_name=logger.name,
    )
