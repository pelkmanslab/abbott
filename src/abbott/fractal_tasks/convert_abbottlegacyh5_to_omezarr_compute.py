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
"""This task converts abbott legacy H5 files to OME-Zarr."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
)
from ngio.images.ome_zarr_container import create_ome_zarr_from_array
from pydantic import Field, validate_call

from abbott.fractal_tasks.converter.io_models import (
    CustomWavelengthInputModel,
    InitArgsCellVoyagerH5toOMEZarr,
    MultiplexingAcquisition,
    OMEZarrBuilderParams,
)
from abbott.fractal_tasks.converter.task_utils import (
    extract_zarr_url_from_h5_filename,
    h5_load,
)

logger = logging.getLogger(__name__)


def convert_single_h5_to_ome(
    zarr_url: str,
    input_file: str,
    level: int,
    acquisitions: dict[str, MultiplexingAcquisition],
    wavelengths: dict[int, str],
    ome_zarr_parameters: OMEZarrBuilderParams,
    metadata: pd.DataFrame,
    masking_label: Optional[str] = None,
    overwrite: bool = False,
):
    """Abbott legacy H5 to OME-Zarr converter task.

    Args:
        zarr_url: Output path to save the OME-Zarr file of the form
            `zarr_dir/plate_name/row/column/`.
        input_file: Path to the H5 file to be converted.
        level: The level of the image to convert. Default is 0.
        acquisitions: Dictionary of acquisitions. Each key is the acquisition
            cycle. Each item defines the
            acquisition by providing allowed image and label channels.
        wavelengths: Dictionary mapping wavelength IDs to their OME-Zarr equivalents.
        ome_zarr_parameters: Parameters for the OME-Zarr builder.
        metadata: Metadata DataFrame containing site metadata.
        masking_label: Optional label for masking ROI e.g. `embryo`.
        overwrite: Whether to overwrite existing converted OME-Zarr files.
    """
    filename = Path(input_file).stem
    logger.info(f"Converting {filename} to OME-Zarr at {zarr_url}")

    # Get metadata
    levels = ome_zarr_parameters.number_multiscale
    xy_scaling_factor = ome_zarr_parameters.xy_scaling_factor
    z_scaling_factor = ome_zarr_parameters.z_scaling_factor

    logger.info(f"Extracting zarr_url from metadata for filename {filename}")
    ROI = extract_zarr_url_from_h5_filename(
        h5_input_path=input_file,
        metadata=metadata,
    )

    # First extract the images from h5 file, then labels
    for c, acquisition in acquisitions.items():
        imgs_dict = {}
        channel_wavelengths = []
        for channel in acquisition.allowed_image_channels:
            try:
                img, scale = h5_load(
                    input_path=input_file,
                    channel=channel,
                    level=level,
                    cycle=int(c),
                    img_type="intensity",
                )
            except RuntimeError as e:
                logger.error(
                    f"Error loading image for channel {channel.label} "
                    f"and wavelength {channel.wavelength_id} in cycle {c}: {e}"
                    "Please check if the channel is present in the H5 file."
                )
            channel_wavelengths.append(wavelengths[channel.wavelength_id])
            channel_label = (
                channel.new_label if channel.new_label is not None else channel.label
            )
            imgs_dict[channel_label] = img

        array = np.stack(list(imgs_dict.values()), axis=0)
        channel_labels = list(imgs_dict.keys())

        # Save for each cycle
        zarr_url_cycle_roi = f"{zarr_url}/{c}/{ROI}"

        xy_pixelsize = float(scale[1])
        z_spacing = float(scale[0]) if len(scale) > 2 else 1

        ome_zarr_container = create_ome_zarr_from_array(
            store=zarr_url_cycle_roi,
            array=array,
            xy_pixelsize=xy_pixelsize,
            z_spacing=z_spacing,
            levels=levels,
            xy_scaling_factor=xy_scaling_factor,
            z_scaling_factor=z_scaling_factor,
            channel_labels=channel_labels,
            channel_wavelengths=channel_wavelengths,
            axes_names=["c", "z", "y", "x"],
            overwrite=overwrite,
        )

        table = ome_zarr_container.build_image_roi_table(f"FOV_{ROI}")
        ome_zarr_container.add_table("FOV_ROI_table", table=table, overwrite=overwrite)

        # Add label images if available
        if acquisition.allowed_label_channels is not None:
            for label_channel in acquisition.allowed_label_channels:
                try:
                    label_img, _ = h5_load(
                        input_path=input_file,
                        channel=label_channel,
                        level=level,
                        cycle=int(c),
                        img_type="label",
                    )
                except RuntimeError as e:
                    logger.error(
                        f"Error loading label for channel {label_channel.label} "
                        f"and wavelength {label_channel.wavelength_id} in cycle {c}: "
                        f"{e}"
                        "Please check if the channel is present in the H5 file."
                    )
                label_name = (
                    label_channel.new_label
                    if label_channel.new_label
                    else label_channel.label
                )

                ome_zarr_container.derive_label(
                    name=label_name,
                ).set_array(label_img)

                if masking_label is not None:
                    try:
                        masking_roi_table = ome_zarr_container.build_masking_roi_table(
                            label=masking_label,
                        )
                        ome_zarr_container.add_table(
                            f"{label_name}_ROI_table",
                            table=masking_roi_table,
                            overwrite=overwrite,
                        )

                    except ValueError as e:
                        logger.info(
                            "Error building masking ROI table for label "
                            f"{masking_label}: {e}"
                            " Masking ROI will not be generated."
                        )

    logger.info(f"Created OME-Zarr container for {filename} at {zarr_url}")
    return zarr_url


@validate_call
def convert_abbottlegacyh5_to_omezarr_compute(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: InitArgsCellVoyagerH5toOMEZarr,
    # Core parameters
    level: int = 0,
    wavelengths: CustomWavelengthInputModel = Field(
        title="Wavelengths", default=CustomWavelengthInputModel()
    ),
    axes_names: str = "ZYX",
    ome_zarr_parameters: OMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default=OMEZarrBuilderParams()
    ),
    masking_label: Optional[str] = None,
    overwrite: bool = False,
):
    """Abbott legacy H5 to OME-Zarr converter task.

    Args:
        zarr_url: Output path to save the OME-Zarr file of the form
            `zarr_dir/plate_name/row/column/`.
        init_args: Initialization arguments passed from init task.
        input_path: Input path to the H5 file, or a folder containing H5 files.
        level: The level of the image to convert. Default is 0.
        wavelengths: Wavelength conversion dictionary mapping.
        axes_names: The layout of the image data. Currently only implemented for 'ZYX'.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.
        masking_label: Optional label for masking ROI e.g. `embryo`.
        overwrite: Whether to overwrite existing converted OME-Zarr files.
    """
    logger.info(f"Converting abbott legacy H5 files to OME-Zarr for {zarr_url}")
    logger.info(f"For axes: {axes_names} and level {level}")

    if axes_names != "ZYX":
        raise ValueError(
            f"Unsupported axes names {axes_names}. "
            "Currently only 'ZYX' is supported for abbott legacy H5 to "
            "OME-Zarr conversion."
        )

    wavelength_conversion_dict = {
        wavelength.wavelength_abbott_legacy: wavelength.wavelength_omezarr
        for wavelength in wavelengths.wavelengths
    }

    # Group files by well
    files = init_args.input_files
    files_well = [file for file in files if init_args.well_ID in Path(file).stem]

    site_metadata, _ = parse_yokogawa_metadata(
        mrf_path=init_args.mrf_path,
        mlf_path=init_args.mlf_path,
    )

    image_list_updates = []
    for file in files_well:
        new_zarr_url = convert_single_h5_to_ome(
            zarr_url=zarr_url,
            input_file=file,
            level=level,
            acquisitions=init_args.acquisitions,
            wavelengths=wavelength_conversion_dict,
            ome_zarr_parameters=ome_zarr_parameters,
            metadata=site_metadata,
            masking_label=masking_label,
            overwrite=overwrite,
        )

        logger.info(f"Succesfully converted {file} to {new_zarr_url}")
        image_update = {"zarr_url": new_zarr_url, "types": {"is_3D": True}}
        image_list_updates.append(image_update)

    return {"image_list_updates": image_list_updates}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_abbottlegacyh5_to_omezarr_compute,
        logger_name=logger.name,
    )
