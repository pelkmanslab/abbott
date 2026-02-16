# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Based on:
# Channel registration logic from Shayan Shamipour <shayan.shamipour@mls.uzh.ch>
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates tranformation for image-based registration."""

import logging
import shutil
import time

import itk
import numpy as np
import pandas as pd
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import ImageProcessingIterator
from ngio.tables import GenericTable
from pydantic import validate_call
from skimage.exposure import rescale_intensity

from abbott.registration.conversions import to_itk
from abbott.registration.itk_elastix import register_transform_only
from abbott.registration.utils import IteratorConfiguration

logger = logging.getLogger(__name__)


def _accumulate_images(move_itk_imgs):
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


def registration_function(
    *,
    image_data: np.ndarray,
    ref_channel_id: int,
    channels_align_ids: list[int],
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    pxl_sizes_zyx,
    parameter_files: list[str],
):
    """Wrap Channel Registration call.

    Args:
        image_data (np.ndarray): Input image data
        ref_channel_id (int): Index of the reference channel in the image data.
        channels_align_ids (list[int]): List of indices of the channels to be aligned.
        lower_rescale_quantile (float): Lower quantile for rescaling the image
            intensities before applying registration.
        upper_rescale_quantile (float): Upper quantile for rescaling the image
            intensities before applying registration.
        pxl_sizes_zyx: Pixel sizes in z, y, x order.
        parameter_files (list[str]): List of paths to elastix parameter files.

    Returns:
        np.ndarray: Transformation parameters.
    """
    # Pre-processing

    image_ref = image_data[ref_channel_id, ...]
    image_ref = rescale_intensity(
        image_ref,
        in_range=(
            np.quantile(image_ref, lower_rescale_quantile),
            np.quantile(image_ref, upper_rescale_quantile),
        ),
    )

    # Pixel-wise addition of channels for channels in channels_align
    move_itk_imgs = []

    for channel in channels_align_ids:
        img_acq_x = image_data[channel, ...]

        img_acq_x = rescale_intensity(
            img_acq_x,
            in_range=(
                np.quantile(img_acq_x, lower_rescale_quantile),
                np.quantile(img_acq_x, upper_rescale_quantile),
            ),
        )

        move = to_itk(img_acq_x, scale=tuple(pxl_sizes_zyx))
        move_itk_imgs.append(move)

    accumulated = _accumulate_images(move_itk_imgs)

    #  Calculate the transformation
    ref = to_itk(image_ref, scale=tuple(pxl_sizes_zyx))

    return register_transform_only(ref, accumulated, parameter_files)


@validate_call
def compute_channel_registration_elastix(
    *,
    # Fractal arguments
    zarr_url: str,
    # Core parameters
    reference_wavelength: str,
    parameter_files: list[str],
    iterator_configuration: IteratorConfiguration | None = None,
    level_path: int = 2,
    lower_rescale_quantile: float = 0.0,
    upper_rescale_quantile: float = 0.99,
    output_table_name: str = "Channel_Registration_Transforms",
    overwrite: bool = True,
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
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the image processing iterator.
        level_path: Pyramid level of the image to be used for registration.
            Choose `0` to process at full resolution.
        lower_rescale_quantile: Lower quantile for rescaling the image
            intensities before applying registration. Can be helpful
             to deal with image artifacts. Default is 0.
        upper_rescale_quantile: Upper quantile for rescaling the image
            intensities before applying registration. Can be helpful
            to deal with image artifacts. Default is 0.99.
        output_table_name: Name of the table to be created in the OME-Zarr container
            to store the registration transformations.
            Default is "Channel_Registration_Transforms".
        overwrite: Whether to overwrite existing registration transformations.
            Default is True.

    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Get all wavelength_ids for OME-Zarr container
    channels_align = ome_zarr.wavelength_ids

    # Check if reference wavelength is in the list otherwise throw an error
    if reference_wavelength not in channels_align:
        raise ValueError(
            f"Registration with {reference_wavelength=} can only work if "
            "reference wavelength exists. "
        )

    # Remove the reference channel from the list
    for channel in channels_align:
        if channel == reference_wavelength:
            channels_align.remove(channel)

    # Get reference channel id
    ref_channel_id = ome_zarr.get_channel_idx(wavelength_id=reference_wavelength)

    # Get channel indices for channels to be aligned
    channels_align_ids = [
        int(ome_zarr.get_channel_idx(wavelength_id=ch)) for ch in channels_align
    ]

    # If len(channels_align) == 0, raise an error
    if len(channels_align) == 0:
        raise ValueError(
            "No channels found to perform channel-based registration. "
            "Please verify more than one channel is present in "
            "the acquisition."
        )

    # Derive new image
    channel_reg_path = f"{zarr_url}_tmp"
    output_ome_zarr = ome_zarr.derive_image(
        store=channel_reg_path,
        overwrite=overwrite,
    )
    output_image = output_ome_zarr.get_image(path=str(level_path))

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Create a basic ImageProcessingIterator
    image = ome_zarr.get_image(path=str(level_path))
    logger.info(f"{image=}")
    iterator = ImageProcessingIterator(
        input_image=image,
        output_image=output_image,
        axes_order=["c", "z", "y", "x"],
    )

    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logger.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logger.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logger.info(f"Iterator updated with ROI table: {iterator=}")

    # Core processing loop
    #
    logger.info("Starting processing...")
    run_times = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)
    dict_transform_rois = {}
    for it, (image_data, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        trans = registration_function(
            image_data=image_data,
            ref_channel_id=ref_channel_id,
            channels_align_ids=channels_align_ids,
            lower_rescale_quantile=lower_rescale_quantile,
            upper_rescale_quantile=upper_rescale_quantile,
            pxl_sizes_zyx=image.pixel_size.zyx,
            parameter_files=parameter_files,
        )

        iteration_time = time.time() - start_time
        run_times.append(iteration_time)

        # Convert the transformation to a dictionary format that
        # can be stored in the table
        transform_dict = {}
        for i in range(trans.GetNumberOfParameterMaps()):
            trans_map = dict(trans.GetParameterMap(i))
            transform_dict[i] = trans_map

        dict_transform_rois[writer.roi.name] = transform_dict

        # Only log the progress every logging_step iterations
        if it % logging_step == 0 or it == num_rois - 1:
            avg_time = sum(run_times) / len(run_times)
            logger.info(
                f"Processed ROI {it + 1}/{num_rois} "
                f"(avg time per ROI: {avg_time:.2f} s)"
            )

    # Write table with transformations to OME-Zarr
    rows = []
    for fov, rois in dict_transform_rois.items():
        for transform_num, transform_data in rois.items():
            rows.append(
                {"FOV": fov, "transform_num": transform_num, "data": transform_data}
            )

    table_transform_rois = pd.DataFrame(rows)

    table_out = GenericTable(table_transform_rois)
    ome_zarr.add_table(
        name=output_table_name,
        table=table_out,
        backend="parquet",
        overwrite=overwrite,
    )

    # Remove the temporary derived image - is there a better way to handle this?
    shutil.rmtree(channel_reg_path)

    logger.info(f"Chanel registration params succesfully calculated for {zarr_url}")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=compute_channel_registration_elastix,
        logger_name=logger.name,
    )
