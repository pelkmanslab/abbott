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

"""Task utils for abbott H5 legacy to OME-Zarr converter Fractal task"""

import logging
from pathlib import Path
from typing import Optional

import dask.array as da
import h5py
import numpy as np
import pandas as pd
from fractal_tasks_core.cellvoyager.metadata import read_metadata_files
from fractal_tasks_core.roi import remove_FOV_overlaps
from fractal_tasks_core.roi.v1 import prepare_FOV_ROI_table

from abbott.fractal_tasks.converter.io_models import (
    ConverterOmeroChannel,
)
from abbott.fractal_tasks.converter.tile import OriginDict, Point

logger = logging.getLogger(__name__)


def find_inconsistent_z_field_patterns(
    mrf_path: str,
    mlf_path: str,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> list[str]:
    """Return MLF exclude patterns for fields with inconsistent Z steps across channels.

    Uses the raw MLF data to detect (well, FieldIndex) pairs where channels have
    different numbers of Z planes, then returns CellVoyager filename patterns that
    can be passed as ``exclude_patterns`` to ``parse_yokogawa_metadata`` to skip
    those fields.

    Args:
        mrf_path: Path to MeasurementDetail.mrf.
        mlf_path: Path to MeasurementData.mlf.
        include_patterns: Optional include patterns forwarded to the MLF reader.
        exclude_patterns: Optional exclude patterns forwarded to the MLF reader.

    Returns:
        List of glob patterns (e.g. ``["*G05F0007*"]``) for the bad fields.
    """
    _, mlf_frame, _ = read_metadata_files(
        mrf_path,
        mlf_path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    # Replicate the Z-count consistency check from fractal_tasks_core get_z_steps
    z_counts = (
        mlf_frame[["well_id", "FieldIndex", "ActionIndex", "Ch", "Z"]]
        .set_index(["well_id", "FieldIndex", "ActionIndex", "Ch"])
        .groupby(level=[0, 1, 2, 3])
        .count()["Z"]
        .groupby(level=["well_id", "FieldIndex"])
    )

    bad_fields = []
    for (well, field), group in z_counts:
        if group.max() != group.min():
            bad_fields.append((well, int(field)))

    # CellVoyager filename format: ..._{Well}_T{t}F{field:03d}L...
    # Use anchors on both sides of the field token to avoid partial matches
    # (e.g. F001 must not accidentally match F0010).
    patterns = [f"*_{well}_*F{field:03d}L*" for well, field in bad_fields]
    if patterns:
        logger.warning(
            f"Found {len(bad_fields)} field(s) with inconsistent Z steps across "
            f"channels: {bad_fields}. These will be excluded from metadata parsing."
        )
    return patterns


def parse_filename(filename: str) -> dict[str, str]:
    """Parse image metadata from filename.

    Args:
        filename: Name of the image.

    Returns:
        Metadata dictionary.
    """
    # Remove extension and folder from filename
    filename = Path(filename).stem

    output = {}

    # Split filename into well + x_coords + y_coords
    filename_fields = filename.split("_")
    if len(filename_fields) < 3:
        raise ValueError(f"{filename} not valid")

    # Assign well
    well = filename_fields[0]
    x_coords = filename_fields[1]
    y_coords = filename_fields[2]
    output["well"] = well

    # Split the well_id into row and column after first letter
    row, col = well[0], well[1:]
    x_micrometer = (
        x_coords.split("-")[1] if "-" in x_coords else x_coords.split("+")[1]
    )  # split at - or +
    y_micrometer = (
        y_coords.split("-")[1] if "-" in y_coords else y_coords.split("+")[1]
    )  # split at - or +
    output["row"] = row
    output["col"] = col
    output["x_coords"] = x_micrometer
    output["y_coords"] = y_micrometer

    return output


def _extract_ROI(
    metadata: pd.DataFrame,
    well_id: str,
    x_micrometer: str,
    y_micrometer: str,
) -> int:
    """Extract ROI from metadata DataFrame."""
    try:
        metadata = metadata.loc[well_id]
    except KeyError as err:
        raise ValueError(
            f"Well '{well_id}' not found in the provided Cellvoyager metadata."
        ) from err
    metadata = metadata.reset_index()
    metadata["x_micrometer"] = metadata["x_micrometer"].apply(
        lambda x: abs(round(float(x)))
    )
    metadata["y_micrometer"] = metadata["y_micrometer"].apply(
        lambda y: abs(round(float(y)))
    )
    metadata = metadata[
        (metadata["x_micrometer"] == abs(round(float(x_micrometer))))
        & (metadata["y_micrometer"] == abs(round(float(y_micrometer))))
    ]
    if metadata.empty:
        raise ValueError(
            "No matching metadata found in the provided Cellvoyager metadata."
        )

    return int(metadata["FieldIndex"].values[0])


def extract_ROIs_from_h5_files(
    files_well: list[str],
    metadata: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Extract ROIs from H5 files and return a dictionary of file to ROI mapping."""
    file_roi_dict = {}
    for h5_file in files_well:
        h5_filename = Path(h5_file).stem
        well_id, x, y = h5_filename.split("_")
        x = x.split("-")[1] if "-" in x else x.split("+")[1]
        y = y.split("-")[1] if "-" in y else y.split("+")[1]
        try:
            ROI = _extract_ROI(metadata, well_id, x, y)
        except ValueError:
            logger.warning(
                f"No matching metadata found for {h5_file} "
                f"(well={well_id}, x={x}, y={y}). "
                "This field may have been excluded due to inconsistent Z steps "
                "across channels. Skipping."
            )
            continue
        file_roi_dict[h5_file] = ROI

    if not file_roi_dict:
        raise ValueError(
            f"No H5 files for well '{well_id}' could be matched to the "
            "Cellvoyager metadata. The well may be absent from the metadata "
            "or all its fields may have been excluded due to inconsistent Z steps."
        )

    # Remove all rows that are not in the file_roi_dict.values
    metadata = metadata.loc[well_id]
    metadata = metadata[metadata.index.isin(file_roi_dict.values())]
    metadata.index = pd.MultiIndex.from_product(
        [[well_id], metadata.index], names=["well_id", "FieldIndex"]
    )
    metadata = remove_FOV_overlaps(metadata)
    metadata = metadata.loc[well_id]

    return file_roi_dict, metadata


def extract_ROI_coordinates(
    metadata: pd.DataFrame, ROI: int
) -> tuple[Point, Point, OriginDict]:
    """Extract metadata (coords, ROI)

    from Cellvoyager metadata DataFrame from h5_file name.
    """
    FOV_ROIs_table = prepare_FOV_ROI_table(metadata)

    roi_array = np.squeeze(FOV_ROIs_table[f"FOV_{ROI}"].X)
    roi_array = np.array(roi_array, dtype=float)

    # Extract the coordinates
    pos_x = roi_array[0] / metadata.pixel_size_x[ROI]
    pos_y = roi_array[1] / metadata.pixel_size_y[ROI]
    pos_z = roi_array[2] / metadata.pixel_size_z[ROI]
    size_x = metadata.x_pixel[ROI]
    size_y = metadata.y_pixel[ROI]
    size_z = metadata.z_pixel[ROI]

    top_left = Point(
        x=pos_x,
        y=pos_y,
        z=pos_z,
    )

    bottom_right = Point(
        x=(pos_x + size_x),
        y=(pos_y + size_y),
        z=(pos_z + size_z),
    )

    origin = OriginDict(
        x_micrometer_original=roi_array[0],
        y_micrometer_original=roi_array[1],
    )

    return top_left, bottom_right, origin


def h5_datasets(f: h5py.File, return_names=False, dsets=None) -> list[h5py.Dataset]:
    """Recursively get all datasets from an HDF5 file."""
    if dsets is None:
        dsets = []

    for group_or_dataset_name in f.keys():
        if isinstance(f[group_or_dataset_name], h5py.Group):
            h5_datasets(
                f[group_or_dataset_name], return_names=return_names, dsets=dsets
            )
        elif isinstance(f[group_or_dataset_name], h5py.Dataset):
            if return_names:
                dsets.append(f[group_or_dataset_name].name)
            else:
                dsets.append(f[group_or_dataset_name])
    return dsets


def h5_select(
    f: h5py.File,
    attrs_select: Optional[dict[str, str | int | tuple[str | int, ...]]] = None,
    not_attrs_select: Optional[dict[str, str | int | tuple[str | int, ...]]] = None,
    return_names: bool = False,
) -> h5py.Dataset:
    """Select a dataset (lazily) from an HDF5 file based on attributes."""
    dsets: list[h5py.Dataset] = []
    for dset in h5_datasets(f):
        check: list[bool] = []
        if attrs_select:
            for a in attrs_select:
                if isinstance(attrs_select[a], (tuple | list)):
                    check.append(dset.attrs.get(a) in attrs_select[a])
                else:
                    check.append(dset.attrs.get(a) == attrs_select[a])

        uncheck: list[bool] = []
        if not_attrs_select:
            for b in not_attrs_select:
                if isinstance(not_attrs_select[b] | (tuple | list)):
                    uncheck.append(dset.attrs.get(b) in not_attrs_select[b])
                else:
                    uncheck.append(dset.attrs.get(b) == not_attrs_select[b])

        if all(check) and not any(uncheck):
            if return_names:
                dsets.append(dset.name)
            else:
                dsets.append(dset)
    if len(dsets) > 1:
        logger.warning(
            "Found multiple datasets matching the selection criteria "
            f"for attributes {attrs_select} and "
            f"not attributes {not_attrs_select}. Returning the first one."
        )
    return dsets[0] if dsets else None


def h5_load(
    input_path: str,
    channel: ConverterOmeroChannel,
    level: int,
    cycle: int,
    img_type: str,
    h5_handle: Optional[h5py.File] = None,
):
    """Load a dataset from an HDF5 file based on metadata."""
    if h5_handle is not None:
        f = h5_handle
    else:
        f = h5py.File(input_path, "r")

    dset = h5_select(
        f=f,
        attrs_select={
            "img_type": img_type,
            "cycle": cycle,
            "stain": channel.label,
            "level": level,
        },
    )
    if dset is None:
        raise FileNotFoundError(
            f"Dataset not found for channel {channel.label}, "
            f"wavelength {channel.wavelength_id}, cycle {cycle}, "
            f"level {level}, img_type {img_type}."
        )

    scale = dset.attrs["element_size_um"]
    # Load lazily using Dask
    arr = da.from_array(dset)
    return arr, scale, f  # Return the file handle to close it later


def find_shape(
    bottom_right: list[Point], dask_array: da.Array
) -> tuple[int, int, int, int]:
    """Find the shape of the image."""
    max_r_x = max(bot_r.x for bot_r in bottom_right)
    max_r_y = max(bot_r.y for bot_r in bottom_right)

    shape_x = int(max_r_x)
    shape_y = int(max_r_y)

    shape_c, shape_z, *_ = dask_array.shape
    return shape_c, shape_z, shape_y, shape_x


def find_chunk_shape(
    dask_array: da.Array,
    max_xy_chunk: int = 4096,
    z_chunk: int = 1,
    c_chunk: int = 1,
) -> tuple[int, int, int, int]:
    """Find the chunk shape of the image."""
    shape_c, shape_z, shape_y, shape_x = dask_array.shape
    chunk_y = min(shape_y, max_xy_chunk)
    chunk_x = min(shape_x, max_xy_chunk)
    chunk_z = min(shape_z, z_chunk)
    chunk_c = min(shape_c, c_chunk)
    return chunk_c, chunk_z, chunk_y, chunk_x
