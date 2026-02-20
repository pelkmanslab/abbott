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
"""Computes and applies warpfield registration."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fractal_tasks_core.tasks._zarr_utils import (
    _get_matching_ref_acquisition_path_heuristic,
    _update_well_metadata,
)
from fractal_tasks_core.utils import (
    _split_well_path_image_path,
)
from ngio import open_ome_zarr_container, open_ome_zarr_well
from ngio.images._ome_zarr_container import OmeZarrContainer
from pydantic import validate_call

from abbott.registration.fractal_helper_tasks import (
    get_acquisition_paths,
    get_pad_width,
    pad_to_max_shape,
    unpad_array,
)

logger = logging.getLogger(__name__)


@validate_call
def apply_registration_warpfield(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    reference_acquisition: int = 0,
    level: int = 0,
    output_image_suffix: str = "registered",
    roi_table: str,
    copy_labels: bool = True,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
    overwrite_input: bool = True,
):
    """Apply warpfield registration to images

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
        level: Which resolution level to apply the registration on. Must match
            the level that was used during computation of the registration.
        output_image_suffix: Name of the output image suffix. E.g. "registered".
        roi_table: Name of the ROI table which has been used during computation of
            registration.
            Examples: `FOV_ROI_table` => loop over the field of views,
            `well_ROI_table` => process the whole well as one image.
        copy_labels: Whether to copy the labels from the reference acquisition
            to the new registered image.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            be loaded.
        masking_label_name: Name of the label that will be used for masking.
            If `use_masks=True`, the label image will be used to mask the
            bounding box of the ROI table. If `use_masks=False`, the whole
            bounding box will be loaded.
        overwrite_input: Whether the old image data should be replaced with the
            newly registered image data.

    """
    logger.info(
        f"Running `warpfield_registration` on {zarr_url=}, "
        f"{roi_table=}, {reference_acquisition=}, "
        f", {use_masks=}, {masking_label_name=}, "
        f"Using {overwrite_input=} and {output_image_suffix=}"
    )

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    new_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_image_suffix}"

    # Get the zarr_url for the reference acquisition
    ome_zarr_well = open_ome_zarr_well(well_url)
    acquisition_ids = ome_zarr_well.acquisition_ids

    acq_dict = get_acquisition_paths(ome_zarr_well)
    logger.info(f"{acq_dict=}")

    if reference_acquisition not in acquisition_ids:
        raise ValueError(
            f"{reference_acquisition=} was not one of the available "
            f"acquisitions in {acquisition_ids=} for well {well_url}"
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
    logger.info(f"Using {reference_zarr_url=}")

    # If the reference zarr url is zarr_url, copy data from reference_zarr_url
    # to new_zarr_url and skip the registration.
    # Warpfield registration doesn't fit in GPU memory for large images, so
    # typically level>0 is used for registration, but the original image is
    # level=0)
    if reference_zarr_url == zarr_url:
        logger.info(
            "Skipping registration for the reference acquisition. "
            "Using the original data as registered data."
        )
        ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
        ome_zarr_new = ome_zarr_ref.derive_image(
            store=new_zarr_url,
            ref_path=str(level),
            copy_labels=False,
            copy_tables=True,
            overwrite=True,
        )

        # Get correct images/labels by pixel_size
        pixel_size = ome_zarr_new.get_image(path="0").pixel_size

        # Copy images
        images = ome_zarr_ref.get_image(pixel_size=pixel_size)
        images_new = ome_zarr_new.get_image(pixel_size=pixel_size)
        images_new.set_array(images.get_array(mode="dask"))
        images_new.consolidate()

        # Copy labels
        label_names = ome_zarr_ref.list_labels()
        for label_name in label_names:
            new_label = ome_zarr_new.derive_label(label_name, overwrite=overwrite_input)
            ref_label = ome_zarr_ref.get_label(label_name, pixel_size=pixel_size)
            ref_label = ref_label.get_array(mode="dask")
            new_label.set_array(ref_label)
            new_label.consolidate()

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
            except ValueError as e:
                logger.warning(
                    f"Could not update the well metadata for {zarr_url=} and "
                    f"{new_img_path}: {e}"
                )

        return image_list_updates

    logger.info(
        f"Using {reference_zarr_url=} as the reference acquisition for registration."
    )

    # Open the OME-Zarr containers for both the reference and moving images
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    ome_zarr_mov = open_ome_zarr_container(zarr_url)

    # Masked loading checks
    if use_masks:
        ref_roi_table = ome_zarr_ref.get_masking_roi_table(roi_table)
    else:
        ref_roi_table = ome_zarr_ref.get_table(roi_table)
    if use_masks:
        if ref_roi_table.table_type() != "masking_roi_table":
            logger.warning(
                f"ROI table {roi_table} in reference OME-Zarr is not "
                f"a masking ROI table. Falling back to use_masks=False."
            )
            use_masks = False
        if masking_label_name is None:
            logger.warning(
                "No masking label provided, but use_masks is True. "
                "Falling back to use_masks=False."
            )
            use_masks = False

    ####################
    # Process images
    ####################
    logger.info("Starting to apply warpfield registration to images...")
    write_registered_zarr(
        zarr_url=zarr_url,
        reference_zarr_url=reference_zarr_url,
        new_zarr_url=new_zarr_url,
        level=level,
        roi_table_name=roi_table,
        ome_zarr_mov=ome_zarr_mov,
        use_masks=use_masks,
        masking_label_name=masking_label_name,
    )
    logger.info("Finished applying warpfield registration to images.")

    ####################
    # Process labels
    ####################
    new_ome_zarr = open_ome_zarr_container(new_zarr_url)

    if copy_labels:
        logger.info(
            "Copying labels from the reference acquisition to the new acquisition."
        )

        label_names = ome_zarr_ref.list_labels()
        for label_name in label_names:
            ref_label = ome_zarr_ref.get_label(label_name, path="0")
            new_label = new_ome_zarr.derive_label(
                label_name, ref_image=ref_label, overwrite=overwrite_input
            )
            ref_label_array = ref_label.get_array(mode="dask")
            new_label.set_array(ref_label_array)
            new_label.consolidate()
        logger.info(
            "Finished copying labels from the reference acquisition to "
            "the new acquisition."
        )

    ####################
    # Copy tables
    # 1. Copy all standard ROI tables from the reference acquisition.
    # 2. Give a warning to tables that aren't standard ROI tables from the given
    # acquisition.
    ####################
    logger.info("Copying tables from the reference acquisition to the new acquisition.")

    table_names = ome_zarr_ref.list_tables()
    for table_name in table_names:
        table = ome_zarr_ref.get_table(table_name)
        if (
            table.table_type() == "roi_table"
            or table.table_type() == "masking_roi_table"
        ):
            # Copy ROI tables from the reference acquisition
            new_ome_zarr.add_table(table_name, table, overwrite=overwrite_input)
        else:
            logger.warning(
                f"{zarr_url} contained a table that is not a standard "
                "ROI table. The `Apply Registration Warpfield` task is "
                "best used before additional e.g. feature tables are generated."
            )
            new_ome_zarr.add_table(
                table_name,
                table,
                overwrite=overwrite_input,
            )

    logger.info(
        "Finished copying tables from the reference acquisition to the new acquisition."
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
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=zarr_url, registered=True)]
        )
    else:
        image_list_updates = dict(
            image_list_updates=[
                dict(
                    zarr_url=new_zarr_url, origin=zarr_url, types=dict(registered=True)
                )
            ]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(new_zarr_url)
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

    return image_list_updates


def write_registered_zarr(
    zarr_url: str,
    reference_zarr_url: str,
    new_zarr_url: str,
    level: int,
    roi_table_name: str,
    ome_zarr_mov: OmeZarrContainer,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
):
    """Apply warpfield registration to a Zarr image

    This function loads the image or label data from a zarr array based on the
    ROI bounding-box coordinates and stores them into a new zarr array.
    The new Zarr array has the same shape as the original array, but will have
    0s where the ROI tables don't specify loading of the image data.
    The ROIs loaded from `list_indices` will be written into the
    `list_indices_ref` position, thus performing translational registration if
    the two lists of ROI indices vary.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be used as
            the basis for the new OME-Zarr image.
        reference_zarr_url: Path or url to the individual OME-Zarr image that
            was used as the reference for the registration.
        new_zarr_url: Path or url to the new OME-Zarr image to be written.
        level: Which resolution level to apply the registration on. Must match
            the level that was used during computation of the registration.
        roi_table_name: Name of the ROI table which has been used during
            computation of registration parameters.
        ome_zarr_mov: OME-Zarr container for the moving image to be registered.
        use_masks: If `True` applies masked image loading, otherwise loads the
            whole bounding box of the ROI table.
        masking_label_name: Name of the label that will be used for masking.

    """
    try:
        import cupy as cp
        import warpfield
    except ImportError as e:
        raise ImportError(
            "The `apply_registration_warpfield` task requires GPU. "
        ) from e

    # Get reference OME-Zarr container and images
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    if use_masks:
        ref_roi_table = ome_zarr_ref.get_masking_roi_table(roi_table_name)
    else:
        ref_roi_table = ome_zarr_ref.get_table(roi_table_name)

    # Derive new ome-zarr container from moving image and copy
    # table & label (if use_masks) from reference
    ome_zarr_new = ome_zarr_mov.derive_image(
        store=new_zarr_url,
        ref_path=str(level),
        copy_labels=False,
        copy_tables=False,
        overwrite=True,
    )
    ome_zarr_new.add_table(roi_table_name, table=ref_roi_table)

    # Get correct images/labels by pixel_size
    pixel_size = ome_zarr_new.get_image(path="0").pixel_size

    if use_masks:
        # Get reference masking label
        new_label = ome_zarr_new.derive_label(masking_label_name, overwrite=True)
        ref_masking_label = ome_zarr_ref.get_label(
            masking_label_name, pixel_size=pixel_size
        )
        ref_masking_label = ref_masking_label.get_array(mode="dask")
        new_label.set_array(ref_masking_label)
        new_label.consolidate()

        ref_images = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            pixel_size=pixel_size,
        )
        dtype = ref_images.dtype
        mov_images = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            pixel_size=pixel_size,
        )
        new_images = ome_zarr_new.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            pixel_size=pixel_size,
        )

    else:
        ref_images = ome_zarr_ref.get_image(pixel_size=pixel_size)
        dtype = ref_images.dtype
        mov_images = ome_zarr_mov.get_image(pixel_size=pixel_size)
        new_images = ome_zarr_new.get_image(pixel_size=pixel_size)

    if use_masks:
        roi_table_mov = ome_zarr_mov.get_masking_roi_table(roi_table_name)
        roi_table_ref = ome_zarr_ref.get_masking_roi_table(roi_table_name)
    else:
        roi_table_mov = ome_zarr_mov.get_table(roi_table_name)
        roi_table_ref = ome_zarr_ref.get_table(roi_table_name)

    # TODO: Add sanity checks on the 2 ROI tables:
    # 1. The number of ROIs need to match
    # 2. The size of the ROIs need to match
    # (otherwise, we can't assign them to the reference regions)
    num_ROIs = len(ref_roi_table.rois())
    for i, ref_roi in enumerate(roi_table_ref.rois()):
        logger.info(f"Now applying registration to ROI {i + 1}/{num_ROIs} ")
        ROI_id = ref_roi.name
        mov_roi = [roi for roi in roi_table_mov.rois() if roi.name == ref_roi.name][0]
        # Load registration parameters
        fn_pattern = f"{roi_table_name}_roi_{ROI_id}_lvl_{level}.h5"
        parameter_path = Path(zarr_url) / "registration"
        parameter_file = sorted(parameter_path.glob(fn_pattern))
        if len(parameter_file) > 1:
            raise ValueError(
                "Found multiple warpfield registration json files for "
                f"{fn_pattern} in {parameter_path}. "
                "Please ensure there is only one file per ROI."
            )

        warp_map = warpfield.register.WarpMap.from_h5(parameter_file[0])

        axes_list = mov_images.axes
        if axes_list == ("c", "z", "y", "x"):
            num_channels = mov_images.num_channels
            # Loop over channels
            for ind_ch in range(num_channels):
                if use_masks:
                    # Set Channel to 0, assuming reference channels all have same shape
                    # Avoids error if n channels in ref and mov differ
                    data_ref = ref_images.get_roi_masked(
                        label=int(ROI_id),
                        c=0,
                        mode="dask",
                    ).squeeze()
                    data_mov = mov_images.get_roi_masked(
                        label=int(ROI_id),
                        c=ind_ch,
                        mode="dask",
                    ).squeeze()

                else:
                    data_ref = ref_images.get_roi(
                        roi=ref_roi,
                        c=0,
                    ).squeeze()
                    data_mov = mov_images.get_roi(
                        roi=mov_roi,
                        c=ind_ch,
                    ).squeeze()

                # Pad to the same shape
                max_shape = tuple(
                    max(r, m)
                    for r, m in zip(data_ref.shape, data_mov.shape, strict=False)
                )
                pad_width = get_pad_width(data_ref.shape, max_shape)
                data_mov = pad_to_max_shape(data_mov, max_shape)

                # Check if the expected shape and the actual shape match
                if data_mov.shape != warp_map.mov_shape:
                    raise ValueError(
                        f"Expected shape {warp_map.mov_shape}, "
                        f"got shape {data_mov.shape}"
                    )

                data_mov_reg = warp_map.apply(data_mov)
                data_mov_reg = data_mov_reg.astype(dtype)  # warpfield returns float32
                data_mov_reg = cp.asnumpy(data_mov_reg)

                # Bring back to original shape
                data_mov_reg = unpad_array(data_mov_reg, pad_width)

                if use_masks:
                    new_images.set_roi_masked(
                        label=int(ROI_id),
                        c=ind_ch,
                        patch=data_mov_reg,
                    )

                else:
                    new_images.set_roi(
                        roi=ref_roi,
                        c=ind_ch,
                        patch=data_mov_reg,
                    )
            new_images.consolidate()

        elif axes_list == ("z", "y", "x"):
            if use_masks:
                data_ref = ref_images.get_roi_masked(
                    label=int(ROI_id),
                    mode="dask",
                )
                data_mov = mov_images.get_roi_masked(
                    label=int(ROI_id),
                    mode="dask",
                )

            else:
                data_mov = mov_images.get_roi(
                    roi=mov_roi,
                    mode="dask",
                )

            # Pad to the same shape
            max_shape = tuple(
                max(r, m) for r, m in zip(data_ref.shape, data_mov.shape, strict=False)
            )
            pad_width = get_pad_width(data_ref.shape, max_shape)
            data_mov = pad_to_max_shape(data_mov, max_shape)

            data_mov_reg = warp_map.apply(data_mov)
            data_mov_reg = data_mov_reg.astype(dtype)  # warpfield returns float32
            data_mov_reg = cp.asnumpy(data_mov_reg)

            # Bring back to original shape
            data_mov_reg = unpad_array(data_mov_reg, pad_width)

            if use_masks:
                new_images.set_roi_masked(
                    label=int(ROI_id),
                    patch=data_mov_reg,
                )

            else:
                new_images.set_roi(
                    roi=ref_roi,
                    patch=data_mov_reg,
                )
            new_images.consolidate()

        elif axes_list == ("c", "y", "x"):
            # TODO: Implement cyx case (based on looping over xy case)
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        elif axes_list == ("y", "x"):
            # TODO: Implement yx case
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )
        else:
            raise NotImplementedError(
                "`write_registered_zarr` has not been implemented for "
                f"a zarr with {axes_list=}"
            )

        # Free GPU memory after each ROI
        cp.get_default_memory_pool().free_all_blocks()

    # Remove labels and tables from new_zarr_url
    shutil.rmtree(f"{new_zarr_url}/tables")
    if use_masks:
        shutil.rmtree(f"{new_zarr_url}/labels")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_warpfield,
        logger_name=logger.name,
    )
