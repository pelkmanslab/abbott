# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Secondary seeded segmentation task."""

import logging
import time
from typing import Optional

import anndata as ad
import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    array_to_bounding_box_table,
    check_valid_ROI_indices,
    convert_ROI_table_to_indices,
    create_roi_table_from_df_list,
    find_overlaps_in_ROI_indices,
    is_ROI_table_valid,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.utils import (
    _split_well_path_image_path,
    rescale_datasets,
)
from pydantic import Field, validate_call
from skimage.segmentation import watershed

from abbott.fractal_tasks.conversions import to_itk, to_labelmap, to_numpy
from abbott.registration.itk_image import apply_image_filter, median
from abbott.segmentation.fractal_helper_tasks import masked_loading_wrapper
from abbott.segmentation.io_models import (
    SeededSegmentationChannelInputModel,
    SeededSegmentationParams,
)
from abbott.segmentation.segmentation_utils import (
    SeededSegmentationCustomNormalizer,
    normalize_seeded_segmentation_channel,
)

logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


def segment_ROI(
    x: np.ndarray,
    num_labels_tot: dict[str, int],
    channel: np.ndarray,
    normalize: SeededSegmentationCustomNormalizer = SeededSegmentationCustomNormalizer(),  # noqa: E501
    label_dtype: Optional[np.dtype] = None,
    pxl_sizes_zyx: Optional[tuple[float, float, float]] = None,
    relabeling: bool = True,
    advanced_model_params: SeededSegmentationParams = SeededSegmentationParams(),
) -> np.ndarray:
    """Internal function that runs seeded segmentation for a single ROI.

    Args:
        x: 4D numpy array.
        num_labels_tot: Number of labels already in total image. Used for
            relabeling purposes. Using a dict to have a mutable object that
            can be edited from within the function without having to be passed
            back through the masked_loading_wrapper.
        channel: (Optional) membrane channel for segmentation.
        normalize: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.
        label_dtype: Label images are cast into this `np.dtype`.
        pxl_sizes_zyx: Tuple of pixel sizes.
        relabeling: Whether relabeling based on num_labels_tot is performed.
        advanced_model_params: Advanced seeded segmentation model parameters.
    """
    # Write some debugging info
    logger.info(
        "[segment_ROI] START |" f" x: {type(x)}, {x.shape} |" f" {normalize.norm_type=}"
    )

    channel = normalize_seeded_segmentation_channel(channel, normalize)

    seeds = x
    # Apply morphological filter to label image
    if advanced_model_params.filter_type is not None:
        filter_type = advanced_model_params.filter_type.value
        filter_value = advanced_model_params.filter_value
        seeds = apply_image_filter(
            to_labelmap(seeds, scale=pxl_sizes_zyx),
            filter_type=filter_type,
            radius=filter_value,
        )
        seeds = to_numpy(seeds)

    # Apply median filter to channel image
    if advanced_model_params.filter_radius is not None:
        channel = to_itk(channel, scale=pxl_sizes_zyx)
        channel = median(channel, radius=advanced_model_params.filter_radius)
        channel = to_numpy(channel)

    # Actual labeling
    t0 = time.perf_counter()
    mask = watershed(
        image=channel,
        markers=seeds,
        compactness=advanced_model_params.compactness,
    )

    if mask.ndim == 2:
        # If we get a 2D image, we still return it as a 3D array
        mask = np.expand_dims(mask, axis=0)
    t1 = time.perf_counter()

    # Write some debugging info
    logger.info(
        "[segment_ROI] END   |"
        f" Elapsed: {t1-t0:.3f} s |"
        f" {mask.shape=},"
        f" {mask.dtype=} (then {label_dtype}),"
        f" {np.max(mask)=} |"
    )

    # Shift labels and update relabeling counters
    if relabeling:
        num_labels_roi = np.max(mask)
        mask[mask > 0] += num_labels_tot["num_labels_tot"]
        num_labels_tot["num_labels_tot"] += num_labels_roi

        # Write some logs
        logger.info(f"ROI had {num_labels_roi=}, {num_labels_tot=}")

        # Check that total number of labels is under control
        if num_labels_tot["num_labels_tot"] > np.iinfo(label_dtype).max:
            raise ValueError(
                "ERROR in re-labeling:"
                f"Reached {num_labels_tot} labels, "
                f"but dtype={label_dtype}"
            )

    return mask.astype(label_dtype)


@validate_call
def seeded_segmentation(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    level: int,
    reference_acquisition: Optional[int] = None,
    label_name: str,
    channel: Optional[SeededSegmentationChannelInputModel] = None,
    input_ROI_table: str = "FOV_ROI_table",
    output_ROI_table: Optional[str] = None,
    output_label_name: str = "cells",
    # Seeded segmentation-related arguments
    relabeling: bool = True,
    use_masks: bool = True,
    advanced_model_params: SeededSegmentationParams = Field(
        default_factory=SeededSegmentationParams
    ),
    overwrite: bool = True,
) -> None:
    """Run seeded segmentation on the ROIs of a single OME-Zarr image.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be segmented. Choose `0` to
            process at full resolution.
        reference_acquisition: If provided, the task will only run seeded_segmentation
            for the reference_zarr_url.
        label_name: Name of the label image to be used as input seeds. Expected to
            be in same zarr_url as channel image.
        channel: Channel for segmentation; requires either
            `wavelength_id` (e.g. `A03_C03`) or `label` (e.g. `ECadherin`),
            but not both. Should contain the membrane marker. If no channel is
            provided, seeded segmentation is run with np.zeros as channel input.
            Also contains normalization options. Default is no normalization.
        input_ROI_table: Name of the ROI table over which the task loops to
            apply seeded segmentation. Examples: `FOV_ROI_table` => loop over
            the field of views, `embryo_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image. For seeded segmentation task it is
            recommended to use a ROI table of type `masking_roi_table`.
        output_ROI_table: If provided, a ROI table with that name is created,
            which will contain the bounding boxes of the newly segmented
            labels. ROI tables should have `ROI` in their name.
        output_label_name: Name of the output label image (e.g. `"cells"`).
        relabeling: If `True`, apply relabeling so that label values are
            unique for all objects in the well.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `embryo_ROI_table`).
        advanced_model_params: Advanced model parameters for seeded segmentation.
        overwrite: If `True`, overwrite the task output.
    """
    logger.info(f"Processing {zarr_url=}")

    # Check if only reference_zarr_url should be processed
    if reference_acquisition is not None:
        well_url, _ = _split_well_path_image_path(zarr_url)
        acq_dict = load_NgffWellMeta(well_url).get_acquisition_paths()
        if reference_acquisition not in acq_dict:
            raise ValueError(
                f"{reference_acquisition=} was not one of the available "
                f"acquisitions in {acq_dict=} for well {well_url}"
            )
        ref_path = acq_dict[reference_acquisition][0]
        reference_zarr_url = f"{well_url}/{ref_path}"
        if zarr_url != reference_zarr_url:
            return

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}")
    logger.info(
        f"NGFF image has level-{level} pixel sizes " f"{actual_res_pxl_sizes_zyx}"
    )

    # Load label image
    try:
        label_zyx = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")
    except TypeError:
        return

    # Find channel index
    if channel.wavelength_id is not None or channel.label is not None:
        omero_channel = channel.get_omero_channel(zarr_url)
        if omero_channel:
            ind_channel = omero_channel.index
            # Load ZYX data
            # Workaround for #788: Only load channel index when there is a channel
            # dimension
            if ngff_image_meta.axes_names[0] != "c":
                data_zyx = da.from_zarr(f"{zarr_url}/{level}")
            else:
                data_zyx = da.from_zarr(f"{zarr_url}/{level}")[ind_channel]
            print("Running task with boarder marker channel")
        else:
            return

    else:
        print("No channel provided, running seeded segmentation without boarder marker")
        data_zyx = da.zeros_like(label_zyx, shape=label_zyx.shape)
    # Read ROI table
    ROI_table_path = f"{zarr_url}/tables/{input_ROI_table}"
    ROI_table = ad.read_zarr(ROI_table_path)

    # Perform some checks on the ROI table
    valid_ROI_table = is_ROI_table_valid(table_path=ROI_table_path, use_masks=use_masks)
    if use_masks and not valid_ROI_table:
        logger.info(
            f"ROI table at {ROI_table_path} cannot be used for masked "
            "loading. Set use_masks=False."
        )
        use_masks = False
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # If we are not planning to use masked loading, fail for overlapping ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {input_ROI_table} table have "
                "overlaps, but we are not using masked loading."
            )

    # Rescale datasets (only relevant for level>0)
    # Workaround for #788
    if ngff_image_meta.axes_names[0] != "c":
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
            coarsening_xy=coarsening_xy,
            reference_level=level,
            remove_channel_axis=False,
        )
    else:
        new_datasets = rescale_datasets(
            datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
            coarsening_xy=coarsening_xy,
            reference_level=level,
            remove_channel_axis=True,
        )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.model_dump()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(zarr_url)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(f"Helper function `prepare_label_group` returned {label_group=}")
    current_label_path = f"{zarr_url}/labels/{output_label_name}/0"
    logger.info(f"Output label path: {current_label_path}")
    store = zarr.storage.FSStore(f"{zarr_url}/labels/{output_label_name}/0")
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=overwrite,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} " f"and chunks {data_zyx.chunks}"
    )

    # Initialize other things
    logger.info(f"Start seeded_segmentation task for {zarr_url}")
    logger.info(f"relabeling: {relabeling}")
    logger.info(f"level: {level}")
    logger.info("Total well shape/chunks:")
    logger.info(f"{data_zyx.shape}")
    logger.info(f"{data_zyx.chunks}")

    # Counters for relabeling
    num_labels_tot = {"num_labels_tot": 0}

    # Iterate over ROIs
    num_ROIs = len(list_indices)

    if output_ROI_table:
        bbox_dataframe_list = []

    logger.info(f"Now starting loop over {num_ROIs} ROIs")
    for i_ROI, indices in enumerate(list_indices):
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        # Prepare channel and label input for seeded_segmentation
        label_np = load_region(label_zyx, region, compute=True, return_as_3D=True)
        img_np = load_region(data_zyx, region, compute=True, return_as_3D=True)

        # Prepare keyword arguments for segment_ROI function
        kwargs_segment_ROI = dict(
            num_labels_tot=num_labels_tot,
            channel=img_np,
            normalize=channel.normalize,
            label_dtype=label_dtype,
            pxl_sizes_zyx=actual_res_pxl_sizes_zyx,
            relabeling=relabeling,
            advanced_model_params=advanced_model_params,
        )

        # Prepare keyword arguments for preprocessing function
        preprocessing_kwargs = {}
        if use_masks:
            preprocessing_kwargs = dict(
                region=region,
                current_label_path=current_label_path,
                ROI_table_path=ROI_table_path,
                ROI_positional_index=i_ROI,
                level=level,
            )

        # Call segment_ROI through the masked-loading wrapper, which includes
        # pre/post-processing functions if needed
        new_label_img = masked_loading_wrapper(
            image_array=label_np,
            function=segment_ROI,
            kwargs=kwargs_segment_ROI,
            use_masks=use_masks,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        if output_ROI_table:
            bbox_df = array_to_bounding_box_table(
                new_label_img,
                actual_res_pxl_sizes_zyx,
                origin_zyx=(s_z, s_y, s_x),
            )

            bbox_dataframe_list.append(bbox_df)

        # Compute and store 0-th level to disk
        da.array(new_label_img).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"End seeded_segmentation task for {zarr_url}, " "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )

    logger.info("End building pyramids")

    if output_ROI_table:
        bbox_table = create_roi_table_from_df_list(bbox_dataframe_list)

        # Write to zarr group
        image_group = zarr.group(zarr_url)
        logger.info(
            "Now writing bounding-box ROI table to "
            f"{zarr_url}/tables/{output_ROI_table}"
        )
        table_attrs = {
            "type": "masking_roi_table",
            "region": {"path": f"../labels/{output_label_name}"},
            "instance_key": "label",
        }
        write_table(
            image_group,
            output_ROI_table,
            bbox_table,
            overwrite=overwrite,
            table_attrs=table_attrs,
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=seeded_segmentation,
        logger_name=logger.name,
    )
