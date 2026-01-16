# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Create OME-NGFF zarr group, for multiplexing dataset."""

import os
from pathlib import Path
from typing import Any, Optional

import fractal_tasks_core
from fractal_tasks_core.cellvoyager.filenames import (
    glob_with_multiple_patterns,
)
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
    sanitize_string,
)
from fractal_tasks_core.cellvoyager.wells import generate_row_col_split
from fractal_tasks_core.channels import check_unique_wavelength_ids
from ngio import ImageInWellPath, create_empty_plate
from pydantic import validate_call

from abbott.fractal_tasks.converter.io_models import (
    AllowedH5Extensions,
    ConverterMultiplexingAcquisition,
    InitArgsCellVoyagerH5toOMEZarr,
)
from abbott.fractal_tasks.converter.task_utils import parse_filename

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

import logging

logger = logging.getLogger(__name__)


@validate_call
def convert_abbottlegacyh5_to_omezarr_init(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Core parameters
    input_dir: str,
    acquisitions: dict[str, ConverterMultiplexingAcquisition],
    # Advanced parameters
    include_glob_patterns: Optional[list[str]] = None,
    exclude_glob_patterns: Optional[list[str]] = None,
    h5_extension: AllowedH5Extensions = AllowedH5Extensions.H5,
    plate_name: str = "AssayPlate_Greiner_CELLSTAR655090",
    mrf_path: str,
    mlf_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create OME-NGFF structure and metadata to host a multiplexing dataset.

    This task takes a set of image folders (i.e. different multiplexing
    acquisitions) and build the internal structure and metadata of a OME-NGFF
    zarr group, without actually loading/writing the image data.

    Each element in input_paths should be treated as a different acquisition.

    Args:
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_dir: Input path to the folder containing H5 files to be converted.
        acquisitions: dictionary of acquisitions. Each key is the acquisition
            identifier (normally 0, 1, 2, 3 etc.). Each item defines the
            acquisition by providing the image_dir and the allowed_channels.
        include_glob_patterns: If specified, only parse images with filenames
            that match with all these patterns. Patterns must be defined as in
            https://docs.python.org/3/library/fnmatch.html, Example:
            `image_glob_pattern=["*_B03_*"]` => only process well B03.
            Can interact with exclude_glob_patterns: All included images - all
            excluded images gives the final list of images to process
        exclude_glob_patterns: If specified, exclude any image where the
            filename matches any of the exclusion patterns. Patterns are
            specified the same as for include_glob_patterns.
        num_levels: Number of resolution-pyramid levels. If set to `5`, there
            will be the full-resolution level and 4 levels of downsampled
            images.
        coarsening_xy: Linear coarsening factor between subsequent levels.
            If set to `2`, level 1 is 2x downsampled, level 2 is 4x downsampled
            etc.
        h5_extension: Filename extension of h5 files
            (e.g. `"h5"` or `"hdf5"`).
        plate_name: Name of the plate that was used to acquire the images.
        mrf_path: Provide path to cycle 0 mrf file, typically
            MeasurementDetail.mrf located in the raw image folder containing tif files.
        mlf_path: Same as for mrf, but for the mlf file. Typical name is
            MeasurementData.mlf.
        overwrite: If `True`, overwrite the task output. Default is `False`.

    Returns:
        A metadata dictionary containing important metadata about the OME-Zarr
            plate, the images and some parameters required by downstream tasks
            (like `num_levels`).
    """
    # Checks on the metadata files:
    # 1. Correct file extensions.
    # 2. Files exist.
    if not mrf_path.endswith(".mrf"):
        raise ValueError(f"{mrf_path} does not end with .mrf")
    if not mlf_path.endswith(".mlf"):
        raise ValueError(f"{mlf_path} does not end with .mlf")
    if not os.path.isfile(mrf_path):
        raise ValueError(f"{mrf_path} does not exist.")
    if not os.path.isfile(mlf_path):
        raise ValueError(f"{mlf_path} does not exist.")

    # Preliminary check if FOV-metadata dataframe can be loaded
    try:
        parse_yokogawa_metadata(
            mrf_path,
            mlf_path,
            include_patterns=include_glob_patterns,
            exclude_patterns=exclude_glob_patterns,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to parse Yokogawa metadata from {mrf_path} and {mlf_path}. "
            f"Please check if the files and the glob patterns are valid."
        ) from e

    # Preliminary checks on acquisitions
    # Note that in metadata the keys of dictionary arguments should be
    # strings (and not integers), so that they can be read from a JSON file
    for key, values in acquisitions.items():
        if not isinstance(key, str):
            raise ValueError(f"{acquisitions=} has non-string keys")
        check_unique_wavelength_ids(values.allowed_image_channels)
        if values.allowed_label_channels is not None:
            check_unique_wavelength_ids(values.allowed_label_channels)
        try:
            int(key)
        except ValueError as err:
            raise ValueError("Acquisition dictionary keys need to be integers") from err

    # Check that all channel names are unique across all acquisitions
    channel_labels_images_new = [
        channel.new_label if channel.new_label is not None else channel.label
        for acq in acquisitions.values()
        for channel in acq.allowed_image_channels
    ]
    if channel_labels_images_new:
        assert len(channel_labels_images_new) == len(set(channel_labels_images_new)), (
            "Channel labels must be unique across all acquisitions. "
            f"Found duplicates: {channel_labels_images_new}"
        )

    channel_labels_labels_new = [
        channel.new_label if channel.new_label is not None else channel.label
        for acq in acquisitions.values()
        if acq.allowed_label_channels is not None
        for channel in acq.allowed_label_channels
    ]
    if channel_labels_labels_new:
        assert len(channel_labels_labels_new) == len(set(channel_labels_labels_new)), (
            "Channel labels must be unique across all label channels. "
            f"Found duplicates: {channel_labels_labels_new}"
        )

    acquisitions_sorted = sorted(set(acquisitions.keys()))

    zarr_plate = sanitize_string(plate_name) + ".zarr"
    full_zarr_plate = str(Path(zarr_dir) / zarr_plate)
    logger.info(f"Creating {full_zarr_plate=}")

    ###
    # Identify all wells
    include_patterns = [f"*{h5_extension.value}"]
    exclude_patterns = []
    if include_glob_patterns:
        include_patterns.extend(include_glob_patterns)
    if exclude_glob_patterns:
        exclude_patterns.extend(exclude_glob_patterns)

    input_files = glob_with_multiple_patterns(
        folder=str(input_dir),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    wells = [parse_filename(os.path.basename(fn))["well"] for fn in input_files]
    wells = sorted(set(wells))
    logger.info(f"{wells=}")

    well_rows_columns = generate_row_col_split(wells)
    logger.info(f"{well_rows_columns=}")

    row_list = [row for row, _ in well_rows_columns]
    column_list = [column for _, column in well_rows_columns]
    row_list = sorted(set(row_list))
    column_list = sorted(set(column_list))

    # Create ImageInWellPath objects for each well
    list_of_images = []
    for row, column in well_rows_columns:
        for acq in acquisitions_sorted:
            image = ImageInWellPath(
                path=str(acq),
                row=row,
                column=column,
                acquisition_id=int(acq),
                acquisition_name=f"acquisition_{acq}",
            )
            list_of_images.append(image)
    logger.info(f"{list_of_images=}")

    # Create the plate
    if not overwrite and Path(full_zarr_plate).exists():
        logger.info(f"Skipping creation of {full_zarr_plate} as it already exists.")

    else:
        create_empty_plate(
            full_zarr_plate,
            name=plate_name,
            images=list_of_images,
            overwrite=True,
        )

    parallelization_list = []
    for acq in acquisitions_sorted:
        for row, column in well_rows_columns:
            parallelization_list.append(
                {
                    "zarr_url": (f"{full_zarr_plate}/{row}/{column}/{acq}"),
                    "init_args": InitArgsCellVoyagerH5toOMEZarr(
                        input_files=input_files,
                        acquisition=acquisitions[acq],
                        well_ID=f"{row}{column}",
                        plate_path=zarr_plate,
                        mrf_path=mrf_path,
                        mlf_path=mlf_path,
                        include_glob_patterns=include_glob_patterns,
                        exclude_glob_patterns=exclude_glob_patterns,
                        overwrite=overwrite,
                    ).model_dump(),
                }
            )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_abbottlegacyh5_to_omezarr_init,
        logger_name=logger.name,
    )
