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
"""This task converts abbott-legacy H5 files to OME-Zarr."""

import logging
from pathlib import Path
from typing import Optional

import dask.array as da
import pandas as pd
from fractal_tasks_core.cellvoyager.metadata import (
    parse_yokogawa_metadata,
)
from ngio import RoiPixels, open_ome_zarr_container
from ngio.images.ome_zarr_container import create_empty_ome_zarr
from ngio.tables import RoiTable
from ngio.utils._errors import NgioFileExistsError
from pydantic import Field, validate_call

from abbott.fractal_tasks.converter.io_models import (
    ConverterMultiplexingAcquisition,
    ConverterOMEZarrBuilderParams,
    CustomWavelengthInputModel,
    InitArgsCellVoyagerH5toOMEZarr,
)
from abbott.fractal_tasks.converter.task_utils import (
    extract_ROI_coordinates,
    extract_ROIs_from_h5_files,
    find_chunk_shape,
    find_shape,
    h5_load,
)

logger = logging.getLogger(__name__)


def convert_single_h5_to_ome(
    zarr_url: str,
    files_well: list[str],
    level: int,
    acquisition_id: str,
    acquisition_params: ConverterMultiplexingAcquisition,
    wavelengths: dict[int, str],
    ome_zarr_parameters: ConverterOMEZarrBuilderParams,
    metadata: pd.DataFrame,
    masking_label: Optional[str] = None,
    overwrite: bool = True,
):
    """Abbott legacy H5 to OME-Zarr converter task (memory-optimized)."""
    logger.info(f"Converting {files_well} to OME-Zarr at {zarr_url}")
    zarr_url = zarr_url.rstrip("/")

    # Extract FOV ROIs
    file_roi_dict, metadata = extract_ROIs_from_h5_files(
        files_well=files_well,
        metadata=metadata,
    )

    # Precompute channel labels and wavelengths once (constant across files)
    img_channels = acquisition_params.allowed_image_channels
    lbl_channels = acquisition_params.allowed_label_channels

    channel_labels = [
        ch.new_label if ch.new_label is not None else ch.label for ch in img_channels
    ]
    channel_wavelengths = [wavelengths[ch.wavelength_id] for ch in img_channels]
    n_channels = len(channel_labels)

    # First pass: collect only lightweight metadata and sample shapes/chunks
    logger.info("Collecting metadata and sample shapes...")
    entries: list[dict] = []
    bottom_rights = []
    h5_handles: dict[str, object] = {}
    pixel_sizes_zyx_dict = None
    built_array = False

    for file in files_well:
        ROI_id = file_roi_dict[file]
        # Use first image channel to read shape/scale;
        # reuse the handle for all reads of this file
        ch0 = img_channels[0]
        if not built_array:
            sample_arrays = []
            for ch in img_channels:
                img0, scale, h5_handle = h5_load(
                    input_path=file,
                    channel=ch,
                    level=level,
                    cycle=int(acquisition_id),
                    img_type="intensity",
                )
                sample_arrays.append(img0)
            built_array = True
            sample_array = da.stack(sample_arrays, axis=0)

        else:
            img0, _, h5_handle = h5_load(
                input_path=file,
                channel=ch0,
                level=level,
                cycle=int(acquisition_id),
                img_type="intensity",
            )

        if file not in h5_handles:
            h5_handles[file] = h5_handle

        if pixel_sizes_zyx_dict is None:
            pixel_sizes_zyx_dict = {"z": scale[0], "y": scale[1], "x": scale[2]}
        top_left, bottom_right, origin = extract_ROI_coordinates(
            metadata=metadata, ROI=ROI_id
        )
        bottom_rights.append(bottom_right)

        shape_zyx = img0.shape

        entries.append(
            {
                "file": file,
                "roi_id": ROI_id,
                "top_left": top_left,
                "bottom_right": bottom_right,
                "origin": origin,
                "shape_zyx": shape_zyx,
            }
        )

    # Validate that shapes are consistent across files
    shapes_zyx = {e["shape_zyx"] for e in entries}
    if len(shapes_zyx) != 1:
        raise ValueError(
            f"All files for {files_well} and acquisition {acquisition_id} "
            "must have the same ZYX shape. Check if channels are missing."
        )

    # Compute on-disk shape/chunks/dtype using lightweight dummies
    on_disk_shape = find_shape(
        bottom_right=bottom_rights,
        dask_array=sample_array,
    )
    # Ensure c dimension is correct
    if len(on_disk_shape) == 3:
        on_disk_shape = (n_channels, *on_disk_shape)
    elif len(on_disk_shape) == 4 and on_disk_shape[0] != n_channels:
        on_disk_shape = (n_channels,) + on_disk_shape[1:]

    chunk_shape = find_chunk_shape(
        dask_array=sample_array,
        max_xy_chunk=ome_zarr_parameters.max_xy_chunk,
        z_chunk=ome_zarr_parameters.z_chunk,
        c_chunk=ome_zarr_parameters.c_chunk,
    )

    # Chunk shape should be <= on-disk shape per axis
    chunk_shape = tuple(
        min(c, s) for c, s in zip(chunk_shape, on_disk_shape, strict=True)
    )
    logging.info(f"Chunk shape: {chunk_shape}" f"On-disk shape {on_disk_shape}")

    # Free memory
    del sample_arrays
    del sample_array

    logger.info("Starting to create OME-Zarr container...")
    # Try creating the empty OME-Zarr container
    try:
        ome_zarr_container = create_empty_ome_zarr(
            store=zarr_url,
            shape=on_disk_shape,
            chunks=chunk_shape,
            xy_pixelsize=pixel_sizes_zyx_dict["x"],
            z_spacing=pixel_sizes_zyx_dict["z"],
            levels=ome_zarr_parameters.number_multiscale,
            xy_scaling_factor=ome_zarr_parameters.xy_scaling_factor,
            z_scaling_factor=ome_zarr_parameters.z_scaling_factor,
            channel_labels=channel_labels,
            channel_wavelengths=channel_wavelengths,
            axes_names=("c", "z", "y", "x"),
            overwrite=overwrite,
        )
    except NgioFileExistsError:
        logger.info(
            f"OME-Zarr group already exists at {zarr_url}. "
            "If you want to overwrite it, set `overwrite=True`."
        )
        ome_zarr_container = open_ome_zarr_container(zarr_url)
        im_list_types = {"is_3D": ome_zarr_container.is_3d}
        return im_list_types

    logger.info(f"OME-Zarr container created at {zarr_url}")
    # Create the well ROI table
    well_roi = ome_zarr_container.build_image_roi_table("Well")
    ome_zarr_container.add_table("well_ROI_table", table=well_roi)

    # Write images per ROI (streaming, no global files_dict)
    image = ome_zarr_container.get_image()
    pixel_size = image.pixel_size
    _fov_rois = []

    logger.info("Writing images to OME-Zarr container...")
    for entry in entries:
        s_z, s_y, s_x = entry["shape_zyx"]
        roi_pix = RoiPixels(
            name=f"FOV_{entry['roi_id']}",
            x=int(entry["top_left"].x),
            y=int(entry["top_left"].y),
            z=int(entry["top_left"].z),
            x_length=s_x,
            y_length=s_y,
            z_length=s_z,
            **entry["origin"]._asdict(),
        )
        roi = roi_pix.to_roi(pixel_size=pixel_size)
        _fov_rois.append(roi)

        # For each ROI, write all channels
        imgs = []
        for ch in img_channels:
            img, _, _ = h5_load(
                input_path=entry["file"],
                channel=ch,
                level=level,
                cycle=int(acquisition_id),
                img_type="intensity",
                h5_handle=h5_handles[entry["file"]],
            )
            imgs.append(img)
        patch = (
            da.expand_dims(img, axis=0) if len(imgs) == 1 else da.stack(imgs, axis=0)
        )
        image.set_roi(roi=roi, patch=patch, axes_order=("c", "z", "y", "x"))

    # Build pyramids, set defaults and set FOV table
    image.consolidate()
    ome_zarr_container.set_channel_percentiles(start_percentile=1, end_percentile=99.9)
    table = RoiTable(rois=_fov_rois)
    ome_zarr_container.add_table("FOV_ROI_table", table=table)
    logger.info("Finished writing images to OME-Zarr container.")

    # Set labels if avalable
    if lbl_channels:
        logger.info("Setting labels for OME-Zarr container...")
        roi_table = ome_zarr_container.get_table("FOV_ROI_table")
        lbl_names = [
            ch.new_label if ch.new_label is not None else ch.label
            for ch in lbl_channels
        ]
        # Map ROI id -> file for quick lookup
        roi_to_file = {e["roi_id"]: e["file"] for e in entries}

        for ch, lbl_name in zip(lbl_channels, lbl_names):
            label = ome_zarr_container.derive_label(name=lbl_name, overwrite=overwrite)
            for roi in roi_table.rois():
                roi_id = int(roi.name.split("_")[-1])
                file = roi_to_file[roi_id]
                label_img, _, _ = h5_load(
                    input_path=file,
                    channel=ch,
                    level=level,
                    cycle=int(acquisition_id),
                    img_type="label",
                    h5_handle=h5_handles[file],
                )
                label.set_roi(roi=roi, patch=label_img, axes_order=("z", "y", "x"))
            label.consolidate()
        logger.info("Finished setting labels for OME-Zarr container.")

    # Close HDF5 handles now that all computations are scheduled
    for handle in set(h5_handles.values()):
        try:
            handle.close()
        except Exception:
            pass

    # Build masking ROI table if masking label is provided
    if masking_label is not None:
        logger.info(f"Building masking ROI table for label {masking_label}...")
        lbl_names = (
            [
                ch.new_label if ch.new_label is not None else ch.label
                for ch in lbl_channels
            ]
            if lbl_channels
            else None
        )
        if lbl_names is not None and masking_label in lbl_names:
            try:
                masking_roi_table = ome_zarr_container.build_masking_roi_table(
                    masking_label
                )
                ome_zarr_container.add_table(
                    f"{masking_label}_ROI_table", table=masking_roi_table
                )
                logger.info(f"Built masking ROI table for label {masking_label}")
            except Exception:
                logger.warning(
                    "Failed to build masking ROI table. "
                    "This might be because the label is not present in the data. "
                )

    logger.info(f"Created OME-Zarr container for {files_well} at {zarr_url}")
    im_list_types = {"is_3D": ome_zarr_container.is_3d}
    return im_list_types


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
    ome_zarr_parameters: ConverterOMEZarrBuilderParams = Field(
        title="OME-Zarr Parameters", default=ConverterOMEZarrBuilderParams()
    ),
    masking_label: Optional[str] = None,
):
    """Abbott legacy H5 to OME-Zarr converter task.

    Args:
        zarr_url: Output path to save the OME-Zarr file of the form
            `zarr_dir/plate_name/row/column/`.
        init_args: Initialization arguments passed from init task.
        input_path: Input path to the H5 file, or a folder containing H5 files.
        level: The level of the image to convert. Currently only level 0 is supported.
        wavelengths: Wavelength conversion dictionary mapping.
        axes_names: The layout of the image data. Currently only implemented for 'ZYX'.
        ome_zarr_parameters (OMEZarrBuilderParams): Parameters for the OME-Zarr builder.
        masking_label: Optional label for masking ROI e.g. `embryo`.
    """
    logger.info(f"Converting abbott legacy H5 files to OME-Zarr for {zarr_url}")
    logger.info(f"For axes: {axes_names} and level {level}")

    if level != 0:
        raise ValueError("Currently only level 0 is supported for conversion.")

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

    # Get acquisition metadata
    site_metadata, _ = parse_yokogawa_metadata(
        mrf_path=init_args.mrf_path,
        mlf_path=init_args.mlf_path,
        include_patterns=init_args.include_glob_patterns,
        exclude_patterns=init_args.exclude_glob_patterns,
    )

    acquisition_id = Path(zarr_url).stem

    im_list_types = convert_single_h5_to_ome(
        zarr_url=zarr_url,
        files_well=files_well,
        level=level,
        acquisition_id=acquisition_id,
        acquisition_params=init_args.acquisition,
        wavelengths=wavelength_conversion_dict,
        ome_zarr_parameters=ome_zarr_parameters,
        metadata=site_metadata,
        masking_label=masking_label,
        overwrite=init_args.overwrite,
    )

    logger.info(f"Succesfully converted {files_well} to {zarr_url}")

    plate_attributes = {
        "well": f"{init_args.well_ID}",
        "plate": f"{init_args.plate_path}",
        "acquisition": str(acquisition_id),
    }

    image_update = {
        "zarr_url": zarr_url,
        "types": im_list_types,
        "attributes": plate_attributes,
    }

    return {"image_list_updates": [image_update]}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_abbottlegacyh5_to_omezarr_compute,
        logger_name=logger.name,
    )
