import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug
from fractal_tasks_core.zarr_utils import OverwriteNotAllowedError

from abbott.fractal_tasks.seeded_segmentation import seeded_segmentation
from abbott.segmentation.io_models import (
    FilterType,
    SeededSegmentationChannelInputModel,
    SeededSegmentationCustomNormalizer,
)


@pytest.fixture(scope="function")
def test_data_dir_2d(tmp_path: Path, zenodo_zarr_stardist: list) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    zenodo_zarr_url = zenodo_zarr_stardist[1]
    dest_dir = (tmp_path / "data_2d").as_posix()
    debug(zenodo_zarr_url, dest_dir)
    shutil.copytree(zenodo_zarr_url, dest_dir)
    return dest_dir


@pytest.fixture(scope="function")
def test_data_dir_3d(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "data_3d").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_seeded_segmentation_workflow_3d(test_data_dir_3d):
    # Task-specific arguments
    input_ROI_table = "emb_ROI_table_2_linked"
    label_name = "nuclei"
    output_label_name = "cells"
    reference_acquisition = 2

    zarr_url = f"{test_data_dir_3d}/B/03/0"

    normalize = SeededSegmentationCustomNormalizer(
        norm_type="custom", lower_percentile=1, upper_percentile=99
    )

    channel = SeededSegmentationChannelInputModel(
        label="ECadherin_2",
        normalize=normalize,
    )

    advanced_model_params = dict(
        filter_radius=2,
        compactness=5,
        filter_params=dict(
            filter_type=FilterType.EROSION,
            filter_value=2,
        ),
    )

    seeded_segmentation(
        zarr_url=zarr_url,
        reference_acquisition=reference_acquisition,
        level=4,
        label_name=label_name,
        channel=channel,
        input_ROI_table=input_ROI_table,
        output_label_name=output_label_name,
        relabeling=True,
        use_masks=True,
        advanced_model_params=advanced_model_params,
        overwrite=True,
    )

    # Test with overwrite=False
    with pytest.raises(OverwriteNotAllowedError):
        seeded_segmentation(
            zarr_url=zarr_url,
            level=4,
            label_name=label_name,
            channel=channel,
            input_ROI_table=input_ROI_table,
            output_label_name=output_label_name,
            relabeling=True,
            use_masks=True,
            advanced_model_params=advanced_model_params,
            overwrite=False,
        )

    # Test with not existing label_name
    seeded_segmentation(
        zarr_url=zarr_url,
        level=4,
        label_name="not_existing_label",
        channel=channel,
        input_ROI_table=input_ROI_table,
        output_label_name=output_label_name,
        relabeling=True,
        use_masks=True,
        advanced_model_params=advanced_model_params,
        overwrite=True,
    )

    # Test with no channel & morphological filter
    channel = SeededSegmentationChannelInputModel(
        normalize=normalize,
    )

    advanced_model_params = dict(
        filter_radius=None,
        compactness=5,
        filter_params=dict(
            filter_type=FilterType.EROSION,
            filter_value=None,
        ),
    )

    seeded_segmentation(
        zarr_url=zarr_url,
        level=4,
        label_name=label_name,
        channel=channel,
        input_ROI_table=input_ROI_table,
        output_label_name=output_label_name,
        relabeling=True,
        use_masks=True,
        advanced_model_params=advanced_model_params,
        overwrite=True,
    )

    # Test with use_masks=False
    seeded_segmentation(
        zarr_url=zarr_url,
        level=4,
        label_name=label_name,
        channel=channel,
        input_ROI_table="FOV_ROI_table",
        output_label_name=output_label_name,
        relabeling=True,
        use_masks=False,
        advanced_model_params=advanced_model_params,
        overwrite=True,
    )

    # Test with zarr_url that is not reference_zarr_url
    zarr_url_not_ref = f"{test_data_dir_3d}/B/03/1"
    seeded_segmentation(
        zarr_url=zarr_url_not_ref,
        reference_acquisition=reference_acquisition,
        level=4,
        label_name=label_name,
        channel=channel,
        input_ROI_table=input_ROI_table,
        output_label_name=output_label_name,
        relabeling=True,
        use_masks=True,
        advanced_model_params=advanced_model_params,
        overwrite=True,
    )

    with pytest.raises((zarr.errors.GroupNotFoundError, KeyError)):
        zarr.open_group(f"{zarr_url_not_ref}/labels/{output_label_name}", "r")
