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
from pathlib import Path
from typing import Callable, Optional

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.tables.v1 import MaskingROITableAttrs
from fractal_tasks_core.upscale_array import convert_region_to_low_res, upscale_array

logger = logging.getLogger(__name__)


def _preprocess_input(
    image_array: np.ndarray,
    *,
    region: tuple[slice, ...],
    current_label_path: str,
    ROI_table_path: str,
    ROI_positional_index: int,
) -> np.ndarray:
    """Preprocess a three-dimensional input to elastix registration task.

    **NOTE**: Adapted from cellpose_segmentation task together with
    masked_loading_wrapper_registration and _postprocess_output to make sure
    e.g. embryos with overlapping bounding boxes are not overwriting
    each other during the writing to zarr step.

    This involves :

    - Loading the masking label array for the appropriate ROI;
    - Extracting the appropriate label value from the `ROI_table.obs`
      dataframe;
    - Constructing the background mask, where the masking label matches with a
      specific label value;
    - Setting the background of `image_array` to `0`;
    - Loading the array which will be needed in postprocessing to restore
      background.

    **NOTE 1**: This function relies on V1 of the Fractal table specifications,
    see
    https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/.

    **NOTE 2**: The pre/post-processing functions and the
    masked_loading_wrapper are currently meant to work as part of the
    cellpose_segmentation task, with the plan of then making them more
    flexible; see
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/340.

    Naming of variables refers to a two-steps labeling, as in "first identify
    organoids, then look for nuclei inside each organoid") :

    - `"masking"` refers to the labels that are used to identify the object
      vs background (e.g. the organoid labels); these labels already exist.
    - `"current"` refers to the labels that are currently being computed in
      the `cellpose_segmentation` task, e.g. the nuclear labels.

    Args:
        image_array: The 3D ZYX array with image data for a specific ROI.
        region: The ZYX indices of the ROI, in a form like
            `(slice(0, 1), slice(1000, 2000), slice(1000, 2000))`.
        current_label_path: Path to the image used as current label, in a form
            like `/somewhere/plate.zarr/A/01/0/labels/nuclei_in_organoids/0`.
        ROI_table_path: Path of the AnnData table for the masking-label ROIs;
            this is used (together with `ROI_positional_index`) to extract
            `label_value`.
        ROI_positional_index: Index of the current ROI, which is used to
            extract `label_value` from `ROI_table_obs`.

    Returns:
        A tuple with three arrays: the preprocessed image array, the background
            mask, the current label.
    """
    logger.info(f"[_preprocess_input] {image_array.shape=}")
    logger.info(f"[_preprocess_input] {region=}")

    # Check that image data are 3D (ZYX)
    if not image_array.ndim == 3:
        raise ValueError(
            "_preprocess_input requires a 3D "
            f"image_array argument, but {image_array.shape=}"
        )

    # Load the ROI table and its metadata attributes
    ROI_table = ad.read_zarr(ROI_table_path)
    attrs = zarr.group(ROI_table_path).attrs
    logger.info(f"[_preprocess_input] {ROI_table_path=}")
    logger.info(f"[_preprocess_input] {attrs.asdict()=}")
    MaskingROITableAttrs(**attrs.asdict())
    label_relative_path = attrs["region"]["path"]
    column_name = attrs["instance_key"]

    # Check that ROI_table.obs has the right column and extract label_value
    if column_name not in ROI_table.obs.columns:
        raise ValueError(
            'In _preprocess_input, "{column_name}" '
            f" missing in {ROI_table.obs.columns=}"
        )
    label_value = int(float(ROI_table.obs[column_name].iloc[ROI_positional_index]))

    # Load masking-label array (lazily)
    masking_label_path = str(Path(ROI_table_path).parent / label_relative_path / "0")
    logger.info(f"{masking_label_path=}")
    masking_label_array = da.from_zarr(masking_label_path)
    logger.info(
        f"[_preprocess_input] {masking_label_path=}, " f"{masking_label_array.shape=}"
    )

    # Load current-label array (lazily)
    current_label_array = da.from_zarr(current_label_path)
    logger.info(
        f"[_preprocess_input] {current_label_path=}, " f"{current_label_array.shape=}"
    )

    # Load ROI data for current label array
    current_label_region = current_label_array[region].compute()

    # Load ROI data for masking label array, with or without upscaling
    if masking_label_array.shape != current_label_array.shape:
        logger.info("Upscaling of masking label is needed")
        lowres_region = convert_region_to_low_res(
            highres_region=region,
            highres_shape=current_label_array.shape,
            lowres_shape=masking_label_array.shape,
        )
        masking_label_region = masking_label_array[lowres_region].compute()
        masking_label_region = upscale_array(
            array=masking_label_region,
            target_shape=current_label_region.shape,
        )
    else:
        masking_label_region = masking_label_array[region].compute()

    # Check that all shapes match
    shapes = (
        masking_label_region.shape,
        current_label_region.shape,
        image_array.shape,
    )
    if len(set(shapes)) > 1:
        raise ValueError(
            "Shape mismatch:\n"
            f"{current_label_region.shape=}\n"
            f"{masking_label_region.shape=}\n"
            f"{image_array.shape=}"
        )

    # Compute background mask
    background_3D = masking_label_region != label_value
    if (masking_label_region == label_value).sum() == 0:
        raise ValueError(f"Label {label_value} is not present in the extracted ROI")

    # Set image background to zero
    image_array[background_3D] = 0
    return image_array


def _postprocess_output(
    modified_array: np.ndarray,
    *,
    original_array: np.ndarray,
    current_label_path: str,
    ROI_table_path: str,
    ROI_positional_index: int,
    region: tuple[slice, ...],
) -> np.ndarray:
    """Postprocess elastix registration output.

    **NOTE**: Adapted from cellpose_segmentation task together with
    _preprocess_input and masked_loading_wrapper_registration to make
    sure e.g. embryos with overlapping bounding boxes are not overwriting
    each other during the writing to zarr step.

    Args:
        modified_array: The 3D (ZYX) array with the correct object data and
            wrong background data.
        original_array: The 3D (ZYX) array with the wrong object data and
            correct background data.
        current_label_path: Path to the image used as current label, in a form
            like `/somewhere/plate.zarr/A/01/0/labels/nuclei_in_organoids/0`.
        ROI_table_path: Path of the AnnData table for the masking-label ROIs;
            this is used (together with `ROI_positional_index`) to extract
            `label_value`.
        ROI_positional_index: Index of the current ROI, which is used to
            extract `label_value` from `ROI_table_obs`.
        region: The ZYX indices of the ROI, in a form like
            `(slice(0, 1), slice(1000, 2000), slice(1000, 2000))`.

    Returns:
        The postprocessed array.
    """
    logger.info(f"[_postprocess_input] {modified_array.shape=}")
    logger.info(f"[_postprocess_input] {region=}")

    # Check that image data are 3D (ZYX)
    if not modified_array.ndim == 3:
        raise ValueError(
            "_postprocess_input requires a 3D "
            f"image_array argument, but {modified_array.shape=}"
        )

    # Load the ROI table and its metadata attributes
    ROI_table = ad.read_zarr(ROI_table_path)
    attrs = zarr.group(ROI_table_path).attrs
    logger.info(f"[_preprocess_input] {ROI_table_path=}")
    logger.info(f"[_preprocess_input] {attrs.asdict()=}")
    MaskingROITableAttrs(**attrs.asdict())
    label_relative_path = attrs["region"]["path"]
    column_name = attrs["instance_key"]

    # Check that ROI_table.obs has the right column and extract label_value
    if column_name not in ROI_table.obs.columns:
        raise ValueError(
            'In _preprocess_input, "{column_name}" '
            f" missing in {ROI_table.obs.columns=}"
        )
    label_value = int(float(ROI_table.obs[column_name].iloc[ROI_positional_index]))

    # Load masking-label array (lazily)
    masking_label_path = str(Path(ROI_table_path).parent / label_relative_path / "0")
    logger.info(f"{masking_label_path=}")
    masking_label_array = da.from_zarr(masking_label_path)
    logger.info(
        f"[_preprocess_input] {masking_label_path=}, " f"{masking_label_array.shape=}"
    )

    # Compute background of original_array
    # Load current-label array (lazily)
    current_label_array = da.from_zarr(current_label_path)
    logger.info(
        f"[_preprocess_input] {current_label_path=}, " f"{current_label_array.shape=}"
    )

    # Load ROI data for current label array
    current_label_region = current_label_array[region].compute()

    # Load ROI data for masking label array, with or without upscaling
    if masking_label_array.shape != current_label_array.shape:
        logger.info("Upscaling of masking label is needed")
        lowres_region = convert_region_to_low_res(
            highres_region=region,
            highres_shape=current_label_array.shape,
            lowres_shape=masking_label_array.shape,
        )
        masking_label_region = masking_label_array[lowres_region].compute()
        masking_label_region = upscale_array(
            array=masking_label_region,
            target_shape=current_label_region.shape,
        )
    else:
        masking_label_region = masking_label_array[region].compute()

    # Check that all shapes match
    shapes = (
        masking_label_region.shape,
        current_label_region.shape,
        modified_array.shape,
    )

    if len(set(shapes)) > 1:
        raise ValueError(
            "Shape mismatch:\n"
            f"{current_label_region.shape=}\n"
            f"{masking_label_region.shape=}\n"
            f"{modified_array.shape=}"
        )

    # Compute background mask
    background_3D = masking_label_region != label_value
    if (masking_label_region == label_value).sum() == 0:
        raise ValueError(f"Label {label_value} is not present in the extracted ROI")

    modified_array[background_3D] = original_array[background_3D]

    return modified_array


def masked_loading_wrapper_registration(
    *,
    function: Callable,
    image_array: np.ndarray,
    kwargs: Optional[dict] = None,
    use_masks: bool,
    preprocessing_kwargs: Optional[dict] = None,
    postprocessing_kwargs: Optional[dict] = None,
):
    """Wrap a function with some pre/post-processing functions

    Args:
        function: The callable function to be wrapped.
        image_array: The image array to be preprocessed and then used as
            positional argument for `function`.
        kwargs: Keyword arguments for `function`.
        use_masks: If `False`, the wrapper only calls
            `function(*args, **kwargs)`.
        preprocessing_kwargs: Keyword arguments for the preprocessing function
            (see call signature of `_preprocess_input()`).
        postprocessing_kwargs: Keyword arguments for the postprocessing function
            (see call signature of `_postprocess_output()`).
    """
    # Optional preprocessing
    if use_masks:
        preprocessing_kwargs = preprocessing_kwargs or {}
        image_array = _preprocess_input(image_array, **preprocessing_kwargs)
    # Run function
    kwargs = kwargs or {}
    registered_img = function(image_array, **kwargs)
    # Optional postprocessing
    if use_masks:
        postprocessing_kwargs = postprocessing_kwargs or {}
        registered_img = _postprocess_output(
            modified_array=registered_img, **postprocessing_kwargs
        )
    return registered_img
