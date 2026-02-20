# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Computes and applies elastix registration."""

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

from abbott.registration.conversions import to_itk, to_numpy
from abbott.registration.fractal_helper_tasks import (
    get_acquisition_paths,
    get_pad_width,
    pad_to_max_shape,
    unpad_array,
)
from abbott.registration.itk_elastix import (
    adapt_itk_params,
    apply_transform,
    load_parameter_files,
)

logger = logging.getLogger(__name__)


@validate_call
def apply_registration_elastix(
    *,
    # Fractal parameters
    zarr_url: str,
    # Core parameters
    reference_acquisition: int = 0,
    output_image_suffix: str = "registered",
    roi_table: str,
    copy_labels: bool = True,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
    overwrite_input: bool = True,
):
    """Apply elastix registration to images

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition to register against. Uses the
            OME-NGFF HCS well metadata acquisition keys to find the reference
            acquisition.
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
        f"Running `apply_registration_elastix` on {zarr_url=}, "
        f"{roi_table=} and {reference_acquisition=}. "
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

    # If the reference zarr url is zarr_url, copy data from reference_zarr_url
    # to new_zarr_url and skip the registration.
    if reference_zarr_url == zarr_url:
        logger.info(
            "Skipping registration for the reference acquisition. "
            "Using the original data as registered data."
        )
        if overwrite_input:
            image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])

        else:
            shutil.copytree(zarr_url, new_zarr_url, dirs_exist_ok=True)
            image_list_updates = dict(
                image_list_updates=[
                    dict(
                        zarr_url=new_zarr_url,
                        origin=zarr_url,
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
                "a masking ROI table. Falling back to use_masks=False."
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
    logger.info("Starting to apply elastix registration to images...")
    write_registered_zarr(
        zarr_url=zarr_url,
        reference_zarr_url=reference_zarr_url,
        new_zarr_url=new_zarr_url,
        roi_table_name=roi_table,
        ome_zarr_mov=ome_zarr_mov,
        use_masks=use_masks,
        masking_label_name=masking_label_name,
    )
    logger.info("Finished applying elastix registration to images.")

    ####################
    # Process labels
    ####################
    new_ome_zarr = open_ome_zarr_container(new_zarr_url)

    if copy_labels:
        logger.info(
            "Copying labels from the reference acquisition to the new " "acquisition."
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
            new_ome_zarr.add_table(table_name, table, overwrite=True)
        else:
            logger.warning(
                f"{zarr_url} contained a table that is not a standard "
                "ROI table. The `Apply Registration (elastix)` task is "
                "best used before additional e.g. feature tables are generated."
            )
            new_ome_zarr.add_table(
                table_name,
                table,
                overwrite=True,
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


def write_registered_zarr(
    zarr_url: str,
    reference_zarr_url: str,
    new_zarr_url: str,
    roi_table_name: str,
    ome_zarr_mov: OmeZarrContainer,
    use_masks: bool = False,
    masking_label_name: Optional[str] = None,
):
    """Apply elastix registration to a Zarr image

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
        new_zarr_url: Path or url to the new OME-Zarr image to be written
        roi_table_name: Name of the ROI table which has been used during
            computation of registration.
        ome_zarr_mov: OME-Zarr container of the moving image to be registered.
        use_masks: If `True` applies masked image loading, otherwise loads the
            whole bounding box of the ROI table.
        masking_label_name: Name of the label that will be used for masking.

    """
    # Get reference OME-Zarr container and images
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    if use_masks:
        ref_roi_table = ome_zarr_ref.get_masking_roi_table(roi_table_name)
    else:
        ref_roi_table = ome_zarr_ref.get_table(roi_table_name)

    ome_zarr_new = ome_zarr_mov.derive_image(
        store=new_zarr_url,
        ref_path="0",
        copy_labels=False,
        copy_tables=False,
        overwrite=True,
    )
    ome_zarr_new.add_table(roi_table_name, table=ref_roi_table)

    if use_masks:
        new_label = ome_zarr_new.derive_label(masking_label_name, overwrite=True)
        # Get reference masking label
        ref_masking_label = ome_zarr_ref.get_label(masking_label_name, path="0")
        ref_masking_label = ref_masking_label.get_array(mode="dask")
        new_label.set_array(ref_masking_label)
        new_label.consolidate()

        ref_images = ome_zarr_ref.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )
        mov_images = ome_zarr_mov.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )
        new_images = ome_zarr_new.get_masked_image(
            masking_label_name=masking_label_name,
            masking_table_name=roi_table_name,
            path="0",
        )

    else:
        ref_images = ome_zarr_ref.get_image()
        mov_images = ome_zarr_mov.get_image()
        new_images = ome_zarr_new.get_image()

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
    for ref_roi in roi_table_ref.rois():
        mov_roi = [roi for roi in roi_table_mov.rois() if roi.name == ref_roi.name][0]
        # Load registration parameters
        ROI_id = mov_roi.name
        fn_pattern = f"{roi_table_name}_roi_{ROI_id}_t*.txt"
        parameter_path = Path(zarr_url) / "registration"
        parameter_files = sorted(parameter_path.glob(fn_pattern))
        parameter_object = load_parameter_files([str(x) for x in parameter_files])

        pxl_sizes_zyx = mov_images.pixel_size.zyx
        axes_list = mov_images.axes

        if axes_list == ("c", "z", "y", "x"):
            num_channels = mov_images.num_channels
            # Loop over channels
            for ind_ch in range(num_channels):
                if use_masks:
                    data_ref = ref_images.get_roi_masked(
                        label=int(ROI_id),
                        c=0,
                    ).squeeze()
                    data_mov = mov_images.get_roi_masked(
                        label=int(ROI_id),
                        c=ind_ch,
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

                # Apply the registration
                itk_img = to_itk(data_mov, scale=pxl_sizes_zyx)
                parameter_object_adapted = adapt_itk_params(
                    parameter_object=parameter_object, itk_img=itk_img
                )
                data_mov_reg = to_numpy(
                    apply_transform(itk_img, parameter_object_adapted)
                )
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
                )
                data_mov = mov_images.get_roi_masked(
                    label=int(ROI_id),
                )

            else:
                data_ref = ref_images.get_roi(roi=ref_roi)
                data_mov = mov_images.get_roi(roi=mov_roi)

            # Pad to the same shape
            max_shape = tuple(
                max(r, m) for r, m in zip(data_ref.shape, data_mov.shape, strict=False)
            )
            pad_width = get_pad_width(data_mov.shape, max_shape)
            data_mov = pad_to_max_shape(data_mov, max_shape)

            # Apply the registration
            itk_img = to_itk(data_mov, scale=tuple(pxl_sizes_zyx))
            parameter_object_adapted = adapt_itk_params(
                parameter_object=parameter_object, itk_img=itk_img
            )
            data_mov_reg = to_numpy(apply_transform(itk_img, parameter_object_adapted))
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

    # Remove labels and tables from new_zarr_url
    shutil.rmtree(f"{new_zarr_url}/tables")
    if use_masks:
        shutil.rmtree(f"{new_zarr_url}/labels")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=apply_registration_elastix,
        logger_name=logger.name,
    )
