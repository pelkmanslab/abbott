# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Based on:
# https://github.com/MaksHess/abbott
# Channel registration logic from Shayan Shamipour <shayan.shamipour@mls.uzh.ch>
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Calculates transformation for 3D image-based registration"""

import logging
import shutil
import time
from collections.abc import Sequence

import itk
import numpy as np
from fractal_tasks_core.tasks._zarr_utils import (
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _split_well_path_image_path,
)
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import ImageProcessingIterator
from ngio.images import ChannelSelectionModel
from pydantic import validate_call

from abbott.registration.conversions import to_itk, to_numpy
from abbott.registration.itk_elastix import adapt_itk_params, apply_transform
from abbott.registration.utils import IteratorConfiguration

logger = logging.getLogger(__name__)


def load_parameter_object(parameter_dict: dict) -> itk.ParameterObject:
    """Load one or multiple parameter files into a parameter object.

    Args:
        parameter_dict: Dictionary containing the transforms.

    Returns:
        Parameter object.
    """
    parameter_object = itk.ParameterObject.New()
    for _, transform in parameter_dict.items():
        parameter_object.AddParameterMap(transform)

    return parameter_object


def apply_transformation_function(
    *,
    image_data: np.ndarray,
    ref_channel_id: int,
    channels_align_ids: Sequence[int],
    transform_map: itk.ParameterObject,
    pixel_size_zyx: Sequence[float],
):
    """Wrap Channel Registration call.

    Args:
        image_data (np.ndarray): Input image data
        ref_channel_id (int): Channel id of the reference channel.
        channels_align_ids (Sequence[int]): Channel ids of the channels to be aligned.
        transform_map (itk.ParameterObject): Parameter object containing the transforms
            to be applied to the channels to be aligned.
        pixel_size_zyx (Sequence[float]): Pixel size of the image in z, y, x order.

    Returns:
        np.ndarray: Transformation parameters.
    """
    # Pre-processing

    image_ref = image_data[ref_channel_id, ...]

    itk_image_ref = to_itk(image_ref, scale=pixel_size_zyx)

    # Adapt itk parameters in case e.g. the transforms were
    # calculated on a different pyramid level
    transform_map = adapt_itk_params(
        parameter_object=transform_map,
        itk_img=itk_image_ref,
    )

    for channel_id in channels_align_ids:
        image_align = image_data[channel_id, ...]

        # Apply the transformation to the channel to be aligned
        registered_channel = apply_transform(
            moving=to_itk(image_align, scale=pixel_size_zyx),
            trans=transform_map,
        )

        # Update the registered channel in the output array
        image_data[channel_id, ...] = to_numpy(registered_channel)

    return image_data


@validate_call
def apply_channel_registration_elastix(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    reference_channel: ChannelSelectionModel,
    iterator_configuration: IteratorConfiguration,
    transformation_table_name: str = "Channel_Registration_Transforms",
    copy_labels: bool = True,
    level_path: int = 0,
    output_image_suffix: str = "channels_registered",
    overwrite_input: bool = False,
    use_masks: bool = False,
    masking_label_name: str | None = None,
):
    """Apply channel registration to images using pre-computed transformations.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_channel: Reference channel used during registration computation.
        transformation_table_name (str): Name of the table in which the transformations
            have been stored in preceeding computation task. Defaults to
            "Channel_Registration_Transforms".
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify a ROI
            table (standard or masking) to restrict processing to specific ROIs.
        copy_labels: Whether to copy the labels from the reference acquisition
            to the new registered image.
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        output_image_suffix (str): Name of the output image suffix. Defaults to
            "channels_registered".
        overwrite_input (bool): Whether to overwrite the input zarr file with the new
            registered zarr file. If False, the new registered zarr file will be
            created with output_image_suffix.
        use_masks: If `True`, use masked loading via a masking ROI table.
            Requires `masking_label_name` and a roi_table set in
            `iterator_configuration`. Falls back to `use_masks=False` if
            either is missing or the ROI table is not a masking ROI table.
        masking_label_name: Name of the label image used for masking ROIs,
            e.g. `embryo`. Required when `use_masks=True`.

    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Get reference channel index using ChannelSelectionModel
    if reference_channel.mode == "label":
        ref_channel_id = ome_zarr.get_channel_idx(
            channel_label=reference_channel.identifier
        )
    elif reference_channel.mode == "wavelength_id":
        ref_channel_id = ome_zarr.get_channel_idx(
            wavelength_id=reference_channel.identifier
        )
    else:  # mode == "index"
        ref_channel_id = int(reference_channel.identifier)

    # Get channel indices for channels to be aligned (all channels except reference)
    channels_align_ids = [
        i for i in range(ome_zarr.num_channels) if i != ref_channel_id
    ]

    # Validate masking configuration
    if use_masks:
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False
        elif iterator_configuration.roi_table is None:
            raise ValueError(
                "use_masks=True requires roi_table to be set in iterator_configuration."
            )
        else:
            _roi_tbl = ome_zarr.get_table(iterator_configuration.roi_table)
            if _roi_tbl.table_type() != "masking_roi_table":
                logger.warning(
                    f"ROI table {iterator_configuration.roi_table!r} is not a "
                    "masking ROI table. Falling back to use_masks=False."
                )
                use_masks = False
            else:
                masking_roi_table = ome_zarr.get_masking_roi_table(
                    iterator_configuration.roi_table
                )

    # Derive the new registered image
    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    registered_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_image_suffix}"
    registered_ome_zarr = ome_zarr.derive_image(
        store=registered_zarr_url, overwrite=True
    )

    # Load the transformation table
    transform_table = ome_zarr.get_table(transformation_table_name).dataframe

    # Core processing loop
    logger.info("Starting processing...")
    run_times = []

    if use_masks:
        # Copy the masking label and table to the registered OME-Zarr so that
        # get_masked_image works on the output container too.
        new_label = registered_ome_zarr.derive_label(masking_label_name, overwrite=True)
        ref_masking_label = ome_zarr.get_label(masking_label_name, path="0")
        new_label.set_array(ref_masking_label.get_array(mode="dask"))
        new_label.consolidate()
        registered_ome_zarr.add_table(
            iterator_configuration.roi_table, masking_roi_table, overwrite=True
        )

        masked_image = ome_zarr.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=iterator_configuration.roi_table,
            path=str(level_path),
        )
        registered_masked_image = registered_ome_zarr.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=iterator_configuration.roi_table,
            path=str(level_path),
        )

        rois = masking_roi_table.rois()
        num_rois = len(rois)
        logging_step = max(1, num_rois // 10)

        for it, roi in enumerate(rois):
            image_data = masked_image.get_roi_masked_as_numpy(
                label=int(roi.name),
                axes_order=["c", "z", "y", "x"],
            )

            transform_roi = transform_table[transform_table["FOV"] == str(roi.name)]
            transform_roi_dict = transform_roi["data"].to_dict()
            transform_map = load_parameter_object(transform_roi_dict)

            start_time = time.time()
            registered_data = apply_transformation_function(
                image_data=image_data,
                ref_channel_id=ref_channel_id,
                channels_align_ids=channels_align_ids,
                transform_map=transform_map,
                pixel_size_zyx=masked_image.pixel_size.zyx,
            )

            for ch in range(registered_data.shape[0]):
                registered_masked_image.set_roi_masked(
                    label=int(roi.name),
                    c=ch,
                    patch=registered_data[ch],
                )

            iteration_time = time.time() - start_time
            run_times.append(iteration_time)

            if it % logging_step == 0 or it == num_rois - 1:
                avg_time = sum(run_times) / len(run_times)
                logger.info(
                    f"Processed ROI {it + 1}/{num_rois} "
                    f"(avg time per ROI: {avg_time:.2f} s)"
                )

        registered_masked_image.consolidate()

    else:
        image = ome_zarr.get_image(path=str(level_path))
        logger.info(f"{image=}")
        registered_image = registered_ome_zarr.get_image(path=str(level_path))

        iterator = ImageProcessingIterator(
            input_image=image,
            output_image=registered_image,
            axes_order=["c", "z", "y", "x"],
        )

        # Strict=False means that if there no z axis or z is size 1, it will still work
        iterator = iterator.by_zyx(strict=False)
        logger.info(f"Iterator created: {iterator=}")

        if iterator_configuration.roi_table is not None:
            table = ome_zarr.get_generic_roi_table(
                name=iterator_configuration.roi_table
            )
            logger.info(f"ROI table retrieved: {table=}")
            iterator = iterator.product(table)
            logger.info(f"Iterator updated with ROI table: {iterator=}")

        num_rois = len(iterator.rois)
        logging_step = max(1, num_rois // 10)

        for it, (image_data, writer) in enumerate(iterator.iter_as_numpy()):
            transform_roi = transform_table[
                transform_table["FOV"] == str(writer.roi.name)
            ]
            transform_roi_dict = transform_roi["data"].to_dict()
            transform_map = load_parameter_object(transform_roi_dict)

            start_time = time.time()
            registered_data = apply_transformation_function(
                image_data=image_data,
                ref_channel_id=ref_channel_id,
                channels_align_ids=channels_align_ids,
                transform_map=transform_map,
                pixel_size_zyx=image.pixel_size.zyx,
            )

            writer(registered_data)
            iteration_time = time.time() - start_time
            run_times.append(iteration_time)

            if it % logging_step == 0 or it == num_rois - 1:
                avg_time = sum(run_times) / len(run_times)
                logger.info(
                    f"Processed ROI {it + 1}/{num_rois} "
                    f"(avg time per ROI: {avg_time:.2f} s)"
                )

    # Copy labels and tables from the original OME-Zarr to the new registered OME-Zarr
    # if overwrite_input is False.
    if not overwrite_input:
        if copy_labels:
            logger.info(
                "Copying labels from non-channel registered OME-Zarr to channel "
                "registered OME-Zarr."
            )

            label_names = ome_zarr.list_labels()
            for label_name in label_names:
                new_label = registered_ome_zarr.derive_label(label_name, overwrite=True)
                ref_label = ome_zarr.get_label(label_name, path="0")
                ref_label_da = ref_label.get_array(mode="dask")
                new_label.set_array(ref_label_da)
                new_label.consolidate()
            logger.info("Finished copying labels.")

        # Copy tables
        logger.info(
            "Copying tables from non-channel registered OME-Zarr to channel "
            "registered OME-Zarr."
        )

        table_names = ome_zarr.list_tables()
        for table_name in table_names:
            table = ome_zarr.get_table(table_name)
            table_type = table.table_type()
            if table_type in ["roi_table", "masking_roi_table"]:
                # Copy ROI tables from the reference acquisition
                registered_ome_zarr.add_table(table_name, table, overwrite=True)
            else:
                logger.warning(
                    f"{zarr_url} contained a table that is not a standard "
                    "(masking) ROI table. The `Apply Channel Registration (elastix)` "
                    "task is best used before additional e.g. feature tables "
                    "are generated."
                )

            logger.info(
                "Finished copying tables from the reference acquisition "
                "to the new acquisition."
            )

    if overwrite_input:
        # If overwrite_input is True, we need to delete the old,
        # non-aligned image and rename the new.
        logger.info("Replace original zarr image with the newly created Zarr image")
        registered_array = registered_image.get_as_dask()
        image.set_array(registered_array)
        image.consolidate()
        shutil.rmtree(registered_zarr_url)
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=registered_zarr_url, origin=zarr_url)]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(registered_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError as e:
            logger.warning(
                f"Could not update the well metadata for {zarr_url=} and "
                f"{new_img_path}: {e}"
            )

    logger.info(f"Chanel registration successfully applied for {zarr_url}")
    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_channel_registration_elastix,
        logger_name=logger.name,
    )
