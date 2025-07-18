# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Based on:
# https://github.com/MaksHess/abbott
# Channel registration logic from Shayan Shamipour <shayan.shamipour@mls.uzh.ch>
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates tranformation for image-based registration."""

import logging
from pathlib import Path

import anndata as ad
import dask.array as da
import itk
import numpy as np
import zarr
from fractal_tasks_core.channels import (
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    load_region,
)
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from abbott.fractal_tasks.conversions import to_itk
from abbott.registration.itk_elastix import register_transform_only

logger = logging.getLogger(__name__)


@validate_call
def compute_channel_registration_elastix(
    *,
    # Fractal arguments
    zarr_url: str,
    # Core parameters
    reference_wavelength: str,
    parameter_files: list[str],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    roi_table: str = "FOV_ROI_table",
    level: int = 2,
) -> None:
    """Calculate registration based on images

    This task consists of 3 parts:

    1. Loading the images of a given ROI (=> loop over ROIs)
    2. Calculating ROI-specific similarity transformation by aligning combined
    channels against the reference channel
    3. Storing the calculated transformation in the ROI table

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_wavelength: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        parameter_files: Paths to the elastix parameter files to be used.
            Usually a single parameter file with the transformation class
            SimilarityTransform to compute channel registration.
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
        f"{reference_wavelength=}."
    )

    # Read some parameters from Zarr metadata
    ngff_image_meta = load_NgffImageMeta(str(zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy

    # Read channels from .zattrs
    channels_align: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarr_url
    )

    # Check if reference channel is in the list otherwise throw an error
    if not any(
        reference_wavelength in channel.wavelength_id for channel in channels_align
    ):
        raise ValueError(
            f"Registration with {reference_wavelength=} can only work if "
            "reference wavelength exists. It was not "
            f"found for zarr_url {zarr_url}."
        )

    # Remove the reference channel from the list
    for channel in channels_align:
        if channel.wavelength_id == reference_wavelength:
            channels_align.remove(channel)

    # If len(channels_align) == 0, raise an error
    if len(channels_align) == 0:
        raise ValueError(
            "No channels found to perform channel-based registration. "
            "Please verify more than one channel is present in "
            "the acquisition."
        )

    # Get channel_index via wavelength_id
    channel_ref: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=zarr_url,
        wavelength_id=reference_wavelength,
    )
    channel_index_ref = channel_ref.index

    # Lazily load zarr array
    data_reference_zyx = da.from_zarr(f"{zarr_url}/{level}")[channel_index_ref]

    # Read ROIs
    ROI_table = ad.read_zarr(f"{zarr_url}/tables/{roi_table}")
    logger.info(f"Found {len(ROI_table)} ROIs in {roi_table=} to be processed.")

    # Check that table type of ROI_table_ref is valid. Note that
    # "ngff:region_table" and None are accepted for backwards compatibility
    valid_table_types = [
        "roi_table",
        "masking_roi_table",
        "ngff:region_table",
        None,
    ]
    ROI_table_group = zarr.open_group(
        f"{zarr_url}/tables/{roi_table}",
        mode="r",
    )
    ref_table_attrs = ROI_table_group.attrs.asdict()
    ref_table_type = ref_table_attrs.get("type")
    if ref_table_type not in valid_table_types:
        raise ValueError(
            f"Table '{roi_table}' (with type '{ref_table_type}') is "
            "not a valid ROI table."
        )

    # Read pixel sizes from zarr attributes
    pxl_sizes_zyx_full_res = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )
    check_valid_ROI_indices(list_indices, roi_table)

    num_ROIs = len(list_indices)
    compute = True
    for i_ROI in range(num_ROIs):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} "
            f"with channel {channel_ref} as "
            "reference."
        )
        img_ref = load_region(
            data_zyx=data_reference_zyx,
            region=convert_indices_to_regions(list_indices[i_ROI]),
            compute=compute,
        )

        # Rescale the ref images
        img_ref = rescale_intensity(
            img_ref,
            in_range=(
                np.quantile(img_ref, lower_rescale_quantile),
                np.quantile(img_ref, upper_rescale_quantile),
            ),
        )

        # Pixel-wise addition of channels for channels in channels_align
        move_itk_imgs = []

        for channel in channels_align:
            channel_wavelength_acq_x = channel.wavelength_id

            # Load and rescale channel to align to ref_channel
            channel_align: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=channel_wavelength_acq_x,
            )
            channel_index_acq_x = channel_align.index

            data_alignment_zyx = da.from_zarr(f"{zarr_url}/{level}")[
                channel_index_acq_x
            ]
            img_acq_x = load_region(
                data_zyx=data_alignment_zyx,
                region=convert_indices_to_regions(list_indices[i_ROI]),
                compute=compute,
            )

            img_acq_x = rescale_intensity(
                img_acq_x,
                in_range=(
                    np.quantile(img_acq_x, lower_rescale_quantile),
                    np.quantile(img_acq_x, upper_rescale_quantile),
                ),
            )

            move = to_itk(img_acq_x, scale=tuple(pxl_sizes_zyx))
            move_itk_imgs.append(move)

        accumulated = accumulate_images(move_itk_imgs)

        ##############
        #  Calculate the transformation
        ##############
        ref = to_itk(img_ref, scale=tuple(pxl_sizes_zyx))
        trans = register_transform_only(ref, accumulated, parameter_files)

        # Write transform parameter files
        # TODO: Add overwrite check (it overwrites by default)
        # FIXME: Figure out where to put files
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = trans.GetParameterMap(i)
            # FIXME: Switch from ROI index to ROI names?
            fn = (
                Path(zarr_url)
                / "channel_registration"
                / (f"{roi_table}_roi_{i_ROI}_t{i}.txt")
            )
            fn.parent.mkdir(exist_ok=True, parents=True)
            trans.WriteParameterFile(trans_map, fn.as_posix())


def accumulate_images(move_itk_imgs):
    """Accumulate ITK images with proper type handling."""
    if not move_itk_imgs:
        return None

    # Define image types
    ImageType = type(move_itk_imgs[0])
    dimension = ImageType.GetImageDimension()
    FloatImageType = itk.Image[itk.F, dimension]

    # Convert first image
    cast_filter = itk.CastImageFilter[ImageType, FloatImageType].New()
    cast_filter.SetInput(move_itk_imgs[0])
    cast_filter.Update()
    result = cast_filter.GetOutput()

    # Add remaining images
    for img in move_itk_imgs[1:]:
        # Cast current image
        cast_filter = itk.CastImageFilter[ImageType, FloatImageType].New()
        cast_filter.SetInput(img)
        cast_filter.Update()
        float_img = cast_filter.GetOutput()

        # Add images with explicit template parameters
        add_filter = itk.AddImageFilter[
            FloatImageType, FloatImageType, FloatImageType
        ].New()
        add_filter.SetInput1(result)
        add_filter.SetInput2(float_img)
        add_filter.Update()
        result = add_filter.GetOutput()

    return result


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_channel_registration_elastix,
        logger_name=logger.name,
    )
