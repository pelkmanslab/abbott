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

import h5py
import numpy as np
import pandas as pd

from abbott.fractal_tasks.converter.io_models import (
    OmeroChannel,
)

logger = logging.getLogger(__name__)


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


def extract_zarr_url_from_h5_filename(
    h5_input_path: str, metadata: pd.DataFrame
) -> int:
    """Extracts the Zarr URL from a given HDF5 filename.

    Args:
        h5_input_path: The name of the HDF5 file.
        metadata: A pandas DataFrame containing metadata with well IDs and coordinates.

    Returns:
        int: The ROI index extracted from the metadata.
    """
    h5_filename = Path(h5_input_path).stem

    well_id, x, y = h5_filename.split("_")
    # Split the well_id into row and column after first letter
    x_micrometer = x.split("-")[1] if "-" in x else x.split("+")[1]  # split at - or +
    y_micrometer = y.split("-")[1] if "-" in y else y.split("+")[1]  # split at - or +

    metawell = metadata.loc[well_id].reset_index()
    metawell = metawell[["FieldIndex", "x_micrometer", "y_micrometer"]]
    metawell = metawell.map(lambda x: abs(round(float(x))))
    filtered_metadata = metawell[
        (metawell["x_micrometer"] == abs(round(float(x_micrometer))))
        & (metawell["y_micrometer"] == abs(round(float(y_micrometer))))
    ]
    ROI = (
        filtered_metadata["FieldIndex"].values[0]
        if not filtered_metadata.empty
        else None
    )
    return int(ROI)


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
    """Select a dataset from an HDF5 file based on attributes."""
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
        raise ValueError("Found multiple datasets matching the selection criteria.")
    return dsets[0] if dsets else None


def h5_load(
    input_path: str,
    channel: OmeroChannel,
    level: int,
    cycle: int,
    img_type: str,
) -> tuple[np.ndarray, list[float]]:
    """Load a dataset from an HDF5 file based on metadata."""
    with h5py.File(input_path, "r") as f:
        dset = h5_select(
            f=f,
            attrs_select={
                "img_type": img_type,
                "cycle": cycle,
                "stain": channel.label,
                "wavelength": int(channel.wavelength_id),
                "level": level,
            },
        )
        scale = dset.attrs["element_size_um"]
        return dset[:], scale
