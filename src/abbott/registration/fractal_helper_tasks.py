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
"""Functions to use masked loading of ROIs before/after processing."""

import logging
from collections import defaultdict

import numpy as np
from ngio.hcs._plate import OmeZarrWell

logger = logging.getLogger(__name__)


def get_acquisition_paths(ome_zarr_well: OmeZarrWell) -> dict:
    """Get a dictionary of acquisition ids and their corresponding paths."""
    files = ome_zarr_well.paths()
    acq_dict = defaultdict(list)
    for file in files:
        acq_id = ome_zarr_well.get_image_acquisition_id(file)
        acq_dict[acq_id].append(file)

    return dict(acq_dict)


def pad_to_max_shape(array, target_shape):
    """Pad array to match target shape, handling mixed larger/smaller dimensions."""
    pad_width = []
    for arr_dim, target_dim in zip(array.shape, target_shape, strict=False):
        diff = target_dim - arr_dim
        if diff > 0:  # array dimension is smaller
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        else:  # array dimension is larger or equal
            pad_width.append((0, 0))
    return np.pad(array, pad_width)


def get_pad_width(array_shape, max_shape):
    """Calculate padding width needed to reach max_shape.

    Args:
        array_shape: Current shape of array (z,y,x)
        max_shape: Target shape to pad to (z,y,x)
    """
    pad_width = []
    for arr_dim, target_dim in zip(array_shape, max_shape, strict=False):
        diff = target_dim - arr_dim
        if diff > 0:  # array dimension is smaller
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
        else:  # array dimension is larger or equal
            pad_width.append((0, 0))
    return pad_width


def unpad_array(padded_array: np.ndarray, pad_width: list[tuple]) -> np.ndarray:
    """Unpad array using stored padding widths.

    Args:
        padded_array: Array that was padded
        pad_width: List of tuples (pad_before, pad_after) for each dimension
    """
    slices = []
    for pad_before, pad_after in pad_width:
        if pad_before == 0 and pad_after == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(pad_before, -pad_after if pad_after > 0 else None))
    return padded_array[tuple(slices)]


def histogram_matching(mov: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Perform histogram matching of a 3D array so that its histogram matches

    that of the template.

    Parameters
    ----------
    mov : np.ndarray
        Moving volume.
    ref : np.ndarray
        Reference volume.

    Returns:
    -------
    matched : np.ndarray
        Transformed source with histogram matched to template.
    """
    # Flatten arrays
    source_flat = mov.ravel()
    template_flat = ref.ravel()

    # Get unique pixel values and corresponding counts
    _, bin_idx, s_counts = np.unique(
        source_flat, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    # Compute normalized CDFs
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source_flat.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / template_flat.size

    # Interpolate pixel values from template
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Map source pixels to matched values
    matched = interp_t_values[bin_idx].reshape(mov.shape)

    return matched.astype(mov.dtype)
