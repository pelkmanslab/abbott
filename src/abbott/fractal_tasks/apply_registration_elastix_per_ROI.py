# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Apply part of 3D image-based elastix registration"""

import logging
import os
import shutil
import time
from pathlib import Path

import anndata as ad
import dask.array as da
import fsspec
import itk
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import load_NgffWellMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import (
    check_valid_ROI_indices,
    convert_indices_to_regions,
    convert_ROI_table_to_indices,
    find_overlaps_in_ROI_indices,
    is_ROI_table_valid,
    is_standard_roi_table,
    load_region,
)
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import (
    _get_matching_ref_acquisition_path_heuristic,
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _get_table_path_dict,
    _split_well_path_image_path,
)
from pydantic import validate_call

from abbott.fractal_tasks.conversions import to_itk, to_numpy
from abbott.registration.fractal_helper_tasks import (
    masked_loading_wrapper_registration,
)
from abbott.registration.itk_elastix import apply_transform, load_parameter_files

logger = logging.getLogger(__name__)


@validate_call
def apply_registration_elastix_per_ROI(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    roi_table: str = "emb_ROI_table",
    label_name: str = "emb_linked",
    reference_acquisition: int = 0,
    overwrite_input: bool = False,
    overwrite_output: bool = True,
    use_masks: bool = True,
    # level: int = 0 TODO: expose possibility to apply registration from e.g. level 1
    # upwards (currently always level 0)
):
    """Apply pre-computed elastix registration transforms to a 3D-Image.

    Useful if there are more than one (embryo) ROI in a FOV.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table: Name of the ROI table for which registrations
            have been calculated using the Compute Registration Elastix task.
            Examples: `emb_ROI_table` => loop over each ROI per FOV.
        label_name: Name of the label that is used to mask the image.
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data. Currently default is
            `overwrite_input=False`.
        overwrite_output: Whether pre-existing registered images (which will
            be named "zarr_url" + _registered) should be overwritten by the
            task. Default is True.
        use_masks: use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `emb_ROI_table`).

    """
    logger.info(zarr_url)
    logger.info(
        f"Running `apply_registration_to_image` on {zarr_url=}, "
        f"{roi_table=} and {reference_acquisition=}. "
        f"Using {overwrite_input=}"
    )

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    suffix = "registered"
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{suffix}"

    # Get the zarr_url for the reference acquisition
    acq_dict = load_NgffWellMeta(well_url).get_acquisition_paths()
    logger.info(load_NgffWellMeta(well_url))
    logger.info(acq_dict)
    if reference_acquisition not in acq_dict:
        raise ValueError(
            f"{reference_acquisition=} was not one of the available "
            f"acquisitions in {acq_dict=} for well {well_url}"
        )
    elif len(acq_dict[reference_acquisition]) > 1:
        ref_path = _get_matching_ref_acquisition_path_heuristic(
            acq_dict[reference_acquisition], old_img_path
        )
        logger.warning(
            "Running registration when there are multiple images of the same "
            "acquisition in a well. Using a heuristic to match the reference "
            f"acquisition. Using {ref_path} as the reference image."
        )
    else:
        ref_path = acq_dict[reference_acquisition][0]
    reference_zarr_url = f"{well_url}/{ref_path}"

    # Get acquisition metadata of zarr_url
    curr_acq = get_acquisition_of_zarr_url(well_url, old_img_path)

    # Special handling for the reference acquisition
    # if acq_dict[zarr_url] == reference_zarr_url:
    if curr_acq == reference_acquisition:
        if overwrite_input:
            # If the input is to be overwritten, nothing needs to happen. The
            # reference acquisition stays as is and due to the output type set
            # in the image list, the type of that OME-Zarr is updated
            return
        else:
            # If the input is not overwritten, a copy of the reference
            # OME-Zarr image needs to be created which has the new name & new
            # metadata. It contains the same data as the original reference
            # image.
            generate_copy_of_reference_acquisition(
                zarr_url=zarr_url,
                new_zarr_url=new_zarr_url,
                overwrite=overwrite_output,
            )
            image_list_updates = dict(
                image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
            )
            # Update the metadata of the the well
            well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
            try:
                _update_well_metadata(
                    well_url=well_url,
                    old_image_path=old_img_path,
                    new_image_path=new_img_path,
                )
            except ValueError:
                logger.warning(f"{new_zarr_url} was already listed in well metadata")
            return image_list_updates

    # Load meta data
    level = 0  # TODO: allow other level besides 0
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels
    axes_list = ngff_image_meta.axes_names
    pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    pxl_sizes_zyx_full_res = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    ROI_table_path_ref = f"{reference_zarr_url}/tables/{roi_table}"
    ROI_table_ref = ad.read_zarr(ROI_table_path_ref)

    ROI_table_path_acq = f"{zarr_url}/tables/{roi_table}"
    ROI_table_acq = ad.read_zarr(ROI_table_path_acq)

    # Perform some checks on the ROI table
    valid_ROI_tables = []
    valid_ROI_table_ref = is_ROI_table_valid(
        table_path=ROI_table_path_ref, use_masks=use_masks
    )
    valid_ROI_tables.append(valid_ROI_table_ref)
    valid_ROI_table_acq = is_ROI_table_valid(
        table_path=ROI_table_path_acq, use_masks=use_masks
    )
    valid_ROI_tables.append(valid_ROI_table_acq)

    for valid_ROI_table in valid_ROI_tables:
        if use_masks and not valid_ROI_table:
            logger.info(
                "ROI table cannot be used for masked "
                "loading. Set use_masks=False ..."
            )
            use_masks = False
            continue
    logger.info(f"{use_masks=}")

    # Create list of indices for 3D ROIs
    ref_list_indices = convert_ROI_table_to_indices(
        ROI_table_ref,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )
    check_valid_ROI_indices(ref_list_indices, roi_table)

    list_indices = convert_ROI_table_to_indices(
        ROI_table_acq,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=pxl_sizes_zyx_full_res,
    )
    check_valid_ROI_indices(list_indices, roi_table)

    # If we are not planning to use masked loading, fail if there are
    # overlapping (bounding boxes of) ROIs
    if not use_masks:
        overlap = find_overlaps_in_ROI_indices(list_indices)
        if overlap:
            raise ValueError(
                f"ROI indices created from {roi_table} table have "
                "overlaps, but we are not using masked loading."
            )

    ####################
    # Process images
    ####################

    old_image_group = zarr.open_group(zarr_url, mode="r")
    new_image_group = zarr.group(new_zarr_url)
    new_image_group.attrs.put(old_image_group.attrs.asdict())
    data_array = da.from_zarr(old_image_group[str(level)])

    new_zarr_array = zarr.create(
        shape=data_array.shape,
        chunks=data_array.chunksize,
        dtype=data_array.dtype,
        store=zarr.storage.FSStore(f"{new_zarr_url}/0"),
        overwrite=overwrite_output,
        dimension_separator="/",
    )

    new_zarr_array_da = da.from_zarr(new_zarr_array)

    for i_ROI, indices in enumerate(list_indices):
        # FIXME: Improve sorting to always achieve correct order (above 9 items)
        region = convert_indices_to_regions(indices)
        ref_region = convert_indices_to_regions(ref_list_indices[i_ROI])
        ROI_lbl_id = i_ROI + 1
        fn_pattern = f"{roi_table}_roi_{ROI_lbl_id}_t*.txt"
        parameter_path = Path(zarr_url) / "registration"
        parameter_files = sorted(parameter_path.glob(fn_pattern))
        parameter_object = load_parameter_files([str(x) for x in parameter_files])

        num_channels = data_array.shape[0]

        if axes_list == ["c", "z", "y", "x"]:
            for ind_ch in range(num_channels):
                logger.info(f"Processing ROI index {i_ROI}, channel {ind_ch}")
                # channel_region_in = (slice(ind_ch, ind_ch + 1), region)
                channel_region_out = (
                    slice(ind_ch, ind_ch + 1),
                    *convert_indices_to_regions(ref_list_indices[i_ROI]),
                )
                img = load_region(
                    data_zyx=data_array[ind_ch], region=region, compute=True
                )
                new_img = load_region(
                    data_zyx=new_zarr_array_da[ind_ch], region=ref_region, compute=True
                )

                kwargs_apply_registration_ROI = dict(
                    ref_indices=ref_list_indices[i_ROI],
                    indices=indices,
                    pxl_sizes_zyx=pxl_sizes_zyx,
                    parameter_object=parameter_object,
                )

                preprocessing_kwargs = {}
                postprocessing_kwargs = {}
                if use_masks:
                    preprocessing_kwargs = dict(
                        region=region,
                        current_label_path=f"{zarr_url}/labels/{label_name}/0",
                        ROI_table_path=ROI_table_path_acq,
                        ROI_positional_index=i_ROI,
                    )
                    postprocessing_kwargs = dict(
                        original_array=new_img,
                        current_label_path=f"{reference_zarr_url}/labels/{label_name}/0",
                        ROI_table_path=ROI_table_path_ref,
                        ROI_positional_index=i_ROI,
                        region=ref_region,
                    )

                registered_img = masked_loading_wrapper_registration(
                    image_array=img,
                    function=apply_registration_ROI,
                    kwargs=kwargs_apply_registration_ROI,
                    use_masks=use_masks,
                    preprocessing_kwargs=preprocessing_kwargs,
                    postprocessing_kwargs=postprocessing_kwargs,
                )

                registered_img = np.expand_dims(registered_img, 0)

                da.array(registered_img).to_zarr(
                    url=new_zarr_array,
                    region=channel_region_out,
                    compute=True,
                )

        elif axes_list == ["z", "y", "x"]:
            logger.info(f"Processing ROI index {i_ROI}")

            img = load_region(data_zyx=data_array, region=region, compute=True)
            new_img = load_region(
                data_zyx=new_zarr_array, region=ref_region, compute=True
            )

            kwargs_apply_registration_ROI = dict(
                ref_indices=ref_list_indices[i_ROI],
                indices=indices,
                pxl_sizes_zyx=pxl_sizes_zyx,
                parameter_object=parameter_object,
            )

            preprocessing_kwargs = {}
            postprocessing_kwargs = {}
            if use_masks:
                preprocessing_kwargs = dict(
                    region=region,
                    current_label_path=f"{zarr_url}/labels/{label_name}/0",
                    ROI_table_path=ROI_table_path_acq,
                    ROI_positional_index=i_ROI,
                )
                postprocessing_kwargs = dict(
                    original_array=new_img,
                    current_label_path=f"{reference_zarr_url}/labels/{label_name}/0",
                    ROI_table_path=ROI_table_path_ref,
                    ROI_positional_index=i_ROI,
                    region=ref_region,
                )

            registered_img = masked_loading_wrapper_registration(
                image_array=img,
                function=apply_registration_ROI,
                kwargs=kwargs_apply_registration_ROI,
                use_masks=use_masks,
                preprocessing_kwargs=preprocessing_kwargs,
                postprocessing_kwargs=postprocessing_kwargs,
            )

            da.array(registered_img).to_zarr(
                url=new_zarr_array,
                region=ref_region,
                compute=True,
            )

        elif axes_list == ["c", "y", "x"]:
            # TODO: Implement cyx case (based on looping over xy case)
            raise NotImplementedError(
                "`apply_registration_ROI` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        elif axes_list == ["y", "x"]:
            # TODO: Implement yx case
            raise NotImplementedError(
                "`apply_registration_ROI` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        else:
            raise NotImplementedError(
                "`apply_registration_ROI` has not been implemented for "
                f"a zarr with {axes_list=}"
            )

    logger.info(f"Finished registration for {zarr_url}, " "now building pyramids.")

    # Starting from on-disk highest-resolution data, build and write to
    # disk a pyramid of coarser levels
    build_pyramid(
        zarrurl=new_zarr_url,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_array.chunksize,
        aggregation_function=np.mean,
    )

    ####################
    # Process labels
    ####################
    # TODO: Do we need registration for labels?
    try:
        labels_group = zarr.open_group(f"{zarr_url}/labels", "r")
        label_list = labels_group.attrs["labels"]
        if label_list:
            logger.warning(
                "Skipping registration of labels ... Label registration "
                "has not been implemented."
            )
    except (zarr.errors.GroupNotFoundError, KeyError):
        logger.info("No labels found in the zarr file ... Continuing ...")

    ####################
    # Copy tables
    # 1. Copy all standard ROI tables from the reference acquisition.
    # 2. Copy all tables that aren't standard ROI tables from the given
    # acquisition.
    ####################
    table_dict_reference = _get_table_path_dict(reference_zarr_url)
    table_dict_component = _get_table_path_dict(zarr_url)

    table_dict = {}
    # Define which table should get copied:
    for table in table_dict_reference:
        if is_standard_roi_table(table):
            table_dict[table] = table_dict_reference[table]
    for table in table_dict_component:
        if not is_standard_roi_table(table):
            if reference_zarr_url != zarr_url:
                logger.warning(
                    f"{zarr_url} contained a table that is not a standard "
                    "ROI table. The `Apply Registration To Image task` is "
                    "best used before additional tables are generated. It "
                    f"will copy the {table} from this acquisition without "
                    "applying any transformations. This will work well if "
                    f"{table} contains measurements. But if {table} is a "
                    "custom ROI table coming from another task, the "
                    "transformation is not applied and it will not match "
                    "with the registered image anymore."
                )
            table_dict[table] = table_dict_component[table]

    if table_dict:
        logger.info(f"Processing the tables: {table_dict}")
        new_image_group = zarr.group(new_zarr_url)

        for table in table_dict.keys():
            logger.info(f"Copying table: {table}")
            # Get the relevant metadata of the Zarr table & add it
            # See issue #516 for the need for this workaround
            max_retries = 20
            sleep_time = 5
            current_round = 0
            while current_round < max_retries:
                try:
                    old_table_group = zarr.open_group(table_dict[table], mode="r")
                    current_round = max_retries
                except zarr.errors.GroupNotFoundError:
                    logger.debug(
                        f"Table {table} not found in attempt {current_round}. "
                        f"Waiting {sleep_time} seconds before trying again."
                    )
                    current_round += 1
                    time.sleep(sleep_time)
            # Write the Zarr table
            curr_table = ad.read_zarr(table_dict[table])
            write_table(
                new_image_group,
                table,
                curr_table,
                table_attrs=old_table_group.attrs.asdict(),
                overwrite=True,
            )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        # Potential for race conditions: Every acquisition reads the
        # reference acquisition, but the reference acquisition also gets
        # modified
        # See issue #516 for the details
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(new_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=new_zarr_url, origin=zarr_url)]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ValueError:
            logger.warning(f"{new_zarr_url} was already listed in well metadata")

    return image_list_updates


def apply_registration_ROI(
    image_array: np.ndarray,
    ref_indices: tuple[slice, ...],
    indices: tuple[slice, ...],
    pxl_sizes_zyx: list,
    parameter_object: itk.ParameterObject,
):
    """Apply registration to single (embryo) ROI

    This function pads the img to be registered to the max. of the axes of both
    ref and acq image as during compute_registration_elastix task. The acq will be
    then registered and unpadded to the ref image ROI shape.

    Args:
        image_array: 3D (masked) numpy array of the image to be registered
        ref_indices: Tuple of slices representing the ROI of reference image
        indices: Tuple of slices representing the ROI
        pxl_sizes_zyx: List of pixel sizes in z, y, x
        parameter_object: ITK parameter object containing computed transforms
    """
    # Get max_shape and pad_with
    ref_shape = get_shape_from_indices(ref_indices)
    move_shape = get_shape_from_indices(indices)

    max_shape = tuple(max(r, m) for r, m in zip(ref_shape, move_shape, strict=False))
    pad_width = get_pad_width(ref_shape, max_shape)

    # Pad to same shape as during compute_registration_elastix task
    image_array = np.squeeze(image_array)
    image_array = pad_to_max_shape(image_array, max_shape)

    itk_img = to_itk(
        image_array,
        scale=tuple(pxl_sizes_zyx),
    )

    parameter_object = adapt_itk_params(
        parameter_object=parameter_object,
        itk_img=itk_img,
    )
    registered_roi = apply_transform(
        itk_img,
        parameter_object,
    )

    # Unpad to ref_img ROI shape
    image_array = unpad_array(to_numpy(registered_roi), pad_width)

    return image_array


def get_shape_from_indices(indices):
    """Convert [z_start, z_stop, y_start, y_stop, x_start, x_stop] to shape (z,y,x)

    This function is used to convert the indices of a region to the shape of the
    region. This is needed to retreive the max. shape used during the
    compute_registration_elastix task.

    Args:
        indices: List of indices [z_start, z_stop, y_start, y_stop, x_start, x_stop]

    """
    pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
    shape = tuple(stop - start for start, stop in pairs)
    return shape


def pad_to_max_shape(array, target_shape):
    """Pad array to match target shape, handling mixed larger/smaller dimensions.

    Args:
        array: Numpy array to be padded
        target_shape: Target shape to which the array should be padded

    """
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


def unpad_array(padded_array, pad_width):
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


def adapt_itk_params(parameter_object, itk_img):
    """Updates spacing & size settings in the parameter object

    This is needed to address https://github.com/pelkmanslab/abbott/issues/10
    This ensures that applying the transformation will output an image in the
    input resolution (instead of the transform resolution)

    Args:
        parameter_object: ITK parameter object
        itk_img: ITK image that will be registered

    """
    for i in range(parameter_object.GetNumberOfParameterMaps()):
        itk_spacing = tuple([str(x) for x in itk_img.GetSpacing()])
        itk_size = tuple([str(x) for x in itk_img.GetRequestedRegion().GetSize()])
        parameter_object.SetParameter(i, "Spacing", itk_spacing)
        parameter_object.SetParameter(i, "Size", itk_size)
    return parameter_object


def generate_copy_of_reference_acquisition(
    zarr_url: str,
    new_zarr_url: str,
    overwrite: bool = True,
):
    """Generate a copy of an existing OME-Zarr with all its components

    Args:
        zarr_url: Path to the existing zarr image
        new_zarr_url: Path to the to be created zarr image
        overwrite: Whether to overwrite a preexisting new_zarr_url
    """
    # Get filesystem and paths for source and destination
    source_fs, source_path = fsspec.core.url_to_fs(zarr_url)
    dest_fs, dest_path = fsspec.core.url_to_fs(new_zarr_url)

    # Check if the source exists
    if not source_fs.exists(source_path):
        raise FileNotFoundError(f"The source Zarr URL '{zarr_url}' does not exist.")

    # Handle overwrite option
    if dest_fs.exists(dest_path):
        if overwrite:
            dest_fs.delete(dest_path, recursive=True)
        else:
            raise FileExistsError(
                f"The destination Zarr URL '{new_zarr_url}' already exists."
            )

    # Copy the source to the destination
    source_fs.copy(source_path, dest_path, recursive=True)

    # Verify the copied Zarr structure
    try:
        zarr.open_group(source_fs.get_mapper(source_path), mode="r")
        zarr.open_group(dest_fs.get_mapper(dest_path), mode="r")
    except Exception as e:
        raise RuntimeError(f"Failed to verify the copied Zarr structure: {e}") from e


def get_acquisition_of_zarr_url(well_url, image_name):
    """Get the acquisition of a given zarr_url

    Args:
        well_url: Url of the HCS plate well
        image_name: Name of the acquisition image
    """
    well_meta = load_NgffWellMeta(well_url)
    for image in well_meta.well.images:
        if image.path == image_name:
            return image.acquisition


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_elastix_per_ROI,
        logger_name=logger.name,
    )
