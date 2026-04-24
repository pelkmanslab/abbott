import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug
from ngio import open_ome_zarr_container
from ngio.images import ChannelSelectionModel

from abbott.fractal_tasks.apply_channel_registration_elastix import (
    apply_channel_registration_elastix,
)
from abbott.fractal_tasks.apply_registration_elastix import apply_registration_elastix

# from abbott.fractal_tasks.apply_registration_warpfield import (
#     apply_registration_warpfield,
# )
from abbott.fractal_tasks.compute_channel_registration_elastix import (
    compute_channel_registration_elastix,
)
from abbott.fractal_tasks.compute_registration_elastix import (
    compute_registration_elastix,
)

# from abbott.fractal_tasks.compute_registration_warpfield import (
#     compute_registration_warpfield,
# )
from abbott.fractal_tasks.init_registration_hcs import init_registration_hcs
from abbott.registration.utils import IteratorConfiguration, LabelInputModel


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "registration_data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir, dirs_exist_ok=True)
    return dest_dir


def test_registration_workflow(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_rigid.txt"),
        # str(Path(__file__).parent / "data/params_affine.txt"),
        # str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    ref_wavelength_id = "A01_C01"
    mov_wavelength_id = "A01_C01"
    roi_table = "FOV_ROI_table"
    level = 3
    reference_acquisition = 2
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]

    for param in parallelization_list:
        compute_registration_elastix(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            ref_wavelength_id=ref_wavelength_id,
            mov_wavelength_id=mov_wavelength_id,
            roi_table=roi_table,
            parameter_files=parameter_files,
            use_masks=False,
            masking_label_name=None,
            level=level,
        )

    # Test zarr_url that needs to be registered
    for zarr_url in zarr_urls:
        apply_registration_elastix(
            zarr_url=zarr_url,
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            output_image_suffix="registered",
            use_masks=False,
            masking_label_name=None,
            overwrite_input=False,
        )


def test_registration_workflow_varying_levels(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_rigid.txt"),
        # str(Path(__file__).parent / "data/params_affine.txt"),
        # str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    ref_wavelength_id = "A01_C01"
    mov_wavelength_id = "A01_C01"
    roi_table = "FOV_ROI_table"
    level = 4
    reference_acquisition = 2
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]

    for param in parallelization_list:
        compute_registration_elastix(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            ref_wavelength_id=ref_wavelength_id,
            mov_wavelength_id=mov_wavelength_id,
            roi_table=roi_table,
            parameter_files=parameter_files,
            use_masks=False,
            masking_label_name=None,
            level=level,
        )

    # Test zarr_url that needs to be registered
    for zarr_url in zarr_urls:
        apply_registration_elastix(
            zarr_url=zarr_url,
            roi_table=roi_table,
            reference_acquisition=reference_acquisition,
            output_image_suffix="registered",
            use_masks=False,
            masking_label_name=None,
            overwrite_input=True,
        )


def test_registration_workflow_masked(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_rigid.txt"),
        # str(Path(__file__).parent / "data/params_affine.txt"),
        # str(Path(__file__).parent / "data/bspline_lvl2.txt"),
    ]
    # Task-specific arguments
    ref_wavelength_id = "A01_C01"
    label_name = "emb_linked"
    roi_table = "emb_ROI_table_2_linked"
    level = 0
    reference_acquisition = 2
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    parallelization_list = init_registration_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]

    for param in parallelization_list:
        compute_registration_elastix(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            ref_wavelength_id=ref_wavelength_id,
            roi_table=roi_table,
            parameter_files=parameter_files,
            use_masks=True,
            masking_label_name=label_name,
            level=level,
        )

    # Test zarr_url that needs to be registered
    apply_registration_elastix(
        zarr_url=zarr_urls[1],
        roi_table=roi_table,
        reference_acquisition=reference_acquisition,
        output_image_suffix="registered_masked",
        use_masks=True,
        masking_label_name=label_name,
        overwrite_input=False,
    )


# def test_registration_workflow_warpfield(test_data_dir):
#     # Task-specific arguments
#     wavelength_id = "A01_C01"
#     roi_table = "FOV_ROI_table"
#     level = 0
#     reference_acquisition = 2
#     path_to_registration_recipe = str(Path(__file__).parent / "data/default.yml")
#     zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

#     parallelization_list = init_registration_hcs(
#         zarr_urls=zarr_urls,
#         zarr_dir="",
#         reference_acquisition=reference_acquisition,
#     )["parallelization_list"]

#     for param in parallelization_list:
#         compute_registration_warpfield(
#             zarr_url=param["zarr_url"],
#             init_args=param["init_args"],
#             wavelength_id=wavelength_id,
#             histogram_normalisation=True,
#             path_to_registration_recipe=path_to_registration_recipe,
#             roi_table=roi_table,
#             use_masks=False,
#             masking_label_name=None,
#             level=level,
#         )

#     # Test zarr_url that needs to be registered
#     apply_registration_warpfield(
#         zarr_url=zarr_urls[1],
#         roi_table=roi_table,
#         reference_acquisition=reference_acquisition,
#         output_image_suffix="registered",
#         use_masks=False,
#         masking_label_name=None,
#         overwrite_input=False,
#     )

# def test_registration_workflow_warpfield_masked(test_data_dir):
#     # Task-specific arguments
#     wavelength_id = "A01_C01"
#     label_name = "emb_linked"
#     roi_table = "emb_ROI_table_2_linked"
#     level = 0
#     reference_acquisition = 2
#     path_to_registration_recipe = str(Path(__file__).parent / "data/default.yml")
#     zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

#     parallelization_list = init_registration_hcs(
#         zarr_urls=zarr_urls,
#         zarr_dir="",
#         reference_acquisition=reference_acquisition,
#     )["parallelization_list"]

#     for param in parallelization_list:
#         compute_registration_warpfield(
#             zarr_url=param["zarr_url"],
#             init_args=param["init_args"],
#             wavelength_id=wavelength_id,
#             path_to_registration_recipe=path_to_registration_recipe,
#             roi_table=roi_table,
#             use_masks=True,
#             masking_label_name=label_name,
#             level=level,
#         )

#     # Test zarr_url that needs to be registered
#     for zarr_url in zarr_urls:
#         apply_registration_warpfield(
#             zarr_url=zarr_url,
#             roi_table=roi_table,
#             reference_acquisition=reference_acquisition,
#             level=level,
#             output_image_suffix="registered_masked",
#             use_masks=True,
#             masking_label_name=label_name,
#             overwrite_input=False,
#         )


def test_channel_registration_workflow(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    # Task-specific arguments
    roi_table = "FOV_ROI_table"
    level = 4
    reference_wavelength = "A01_C01"
    zarr_url = f"{test_data_dir}/B/03/0"

    iterator_configuration = IteratorConfiguration(roi_table=roi_table)

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier=reference_wavelength
        ),
        parameter_files=parameter_files,
        iterator_configuration=iterator_configuration,
        level_path=level,
        lower_rescale_quantile=0.0,
        upper_rescale_quantile=0.99,
    )

    # Test zarr_url that needs to be registered
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier=reference_wavelength
        ),
        iterator_configuration=iterator_configuration,
        level_path=4,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_url}_channels_registered"
    zarr.open_group(new_zarr_url, mode="r")

    # Pre-existing output can be overwritten
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier=reference_wavelength
        ),
        iterator_configuration=iterator_configuration,
        level_path=4,
        overwrite_input=True,
    )


def test_channel_registration_workflow_masked(test_data_dir):
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    label_name = "emb_linked"
    roi_table = "emb_ROI_table_2_linked"
    level = 0
    reference_wavelength = "A01_C01"
    zarr_url = f"{test_data_dir}/B/03/0"

    iterator_configuration = IteratorConfiguration(roi_table=roi_table)

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier=reference_wavelength
        ),
        parameter_files=parameter_files,
        iterator_configuration=iterator_configuration,
        level_path=level,
        lower_rescale_quantile=0.0,
        upper_rescale_quantile=0.99,
        use_masks=True,
        masking_label_name=label_name,
    )

    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier=reference_wavelength
        ),
        iterator_configuration=iterator_configuration,
        level_path=level,
        use_masks=True,
        masking_label_name=label_name,
        overwrite_input=False,
    )


def test_channel_registration_masked_fallback_no_label(test_data_dir, caplog):
    """Falls back to non-masked when masking_label_name is not provided."""
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    zarr_url = f"{test_data_dir}/B/03/0"
    iterator_configuration = IteratorConfiguration(roi_table="FOV_ROI_table")

    import logging

    with caplog.at_level(logging.WARNING):
        compute_channel_registration_elastix(
            zarr_url=zarr_url,
            reference_channel=ChannelSelectionModel(
                mode="wavelength_id", identifier="A01_C01"
            ),
            parameter_files=parameter_files,
            iterator_configuration=iterator_configuration,
            level_path=4,
            use_masks=True,
            masking_label_name=None,
        )

    assert "Falling back to use_masks=False" in caplog.text


def test_channel_registration_masked_fallback_non_masking_table(test_data_dir, caplog):
    """Falls back to non-masked when roi_table is not a masking ROI table."""
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    zarr_url = f"{test_data_dir}/B/03/0"
    # FOV_ROI_table is a plain roi_table, not a masking_roi_table
    iterator_configuration = IteratorConfiguration(roi_table="FOV_ROI_table")

    import logging

    with caplog.at_level(logging.WARNING):
        compute_channel_registration_elastix(
            zarr_url=zarr_url,
            reference_channel=ChannelSelectionModel(
                mode="wavelength_id", identifier="A01_C01"
            ),
            parameter_files=parameter_files,
            iterator_configuration=iterator_configuration,
            level_path=4,
            use_masks=True,
            masking_label_name="emb_linked",
        )

    assert "Falling back to use_masks=False" in caplog.text


def test_channel_registration_on_labels(test_data_dir):
    """Channel registration computed on label images instead of intensity images."""
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    roi_table = "FOV_ROI_table"
    level = 0
    zarr_url = f"{test_data_dir}/B/03/0"
    output_table_name = "Channel_Registration_Transforms_labels"

    iterator_configuration = IteratorConfiguration(roi_table=roi_table)

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier="A01_C01"
        ),
        calculate_on_labels=LabelInputModel(
            reference_label_name="nuclei",
            align_label_names=["emb_linked"],
        ),
        parameter_files=parameter_files,
        iterator_configuration=iterator_configuration,
        level_path=level,
        output_table_name=output_table_name,
    )

    ome_zarr = open_ome_zarr_container(zarr_url)
    table = ome_zarr.get_table(output_table_name)
    assert table is not None


def test_channel_registration_on_labels_masked(test_data_dir):
    """Label-based channel registration with masking ROIs."""
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    label_name = "emb_linked"
    roi_table = "emb_ROI_table_2_linked"
    level = 0
    zarr_url = f"{test_data_dir}/B/03/0"
    output_table_name = "Channel_Registration_Transforms_labels_masked"

    iterator_configuration = IteratorConfiguration(roi_table=roi_table)

    compute_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_channel=ChannelSelectionModel(
            mode="wavelength_id", identifier="A01_C01"
        ),
        calculate_on_labels=LabelInputModel(
            reference_label_name="nuclei",
            align_label_names=["emb_linked"],
        ),
        parameter_files=parameter_files,
        iterator_configuration=iterator_configuration,
        level_path=level,
        output_table_name=output_table_name,
        use_masks=True,
        masking_label_name=label_name,
    )

    ome_zarr = open_ome_zarr_container(zarr_url)
    table = ome_zarr.get_table(output_table_name)
    assert table is not None


def test_channel_registration_on_labels_missing_align_label(test_data_dir, caplog):
    """Missing labels in align_label_names are skipped with a warning."""
    import logging

    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    roi_table = "FOV_ROI_table"
    level = 0
    zarr_url = f"{test_data_dir}/B/03/0"
    output_table_name = "Channel_Registration_Transforms_missing_label"

    iterator_configuration = IteratorConfiguration(roi_table=roi_table)

    with caplog.at_level(logging.WARNING):
        compute_channel_registration_elastix(
            zarr_url=zarr_url,
            reference_channel=ChannelSelectionModel(
                mode="wavelength_id", identifier="A01_C01"
            ),
            calculate_on_labels=LabelInputModel(
                reference_label_name="nuclei",
                align_label_names=["emb_linked", "nonexistent_label"],
            ),
            parameter_files=parameter_files,
            iterator_configuration=iterator_configuration,
            level_path=level,
            output_table_name=output_table_name,
        )

    assert "nonexistent_label" in caplog.text
    assert "Skipping" in caplog.text

    ome_zarr = open_ome_zarr_container(zarr_url)
    table = ome_zarr.get_table(output_table_name)
    assert table is not None


def test_channel_registration_on_labels_all_align_labels_missing(test_data_dir):
    """Raises ValueError when all align_label_names are missing."""
    parameter_files = [
        str(Path(__file__).parent / "data/params_similarity_level1.txt"),
    ]
    zarr_url = f"{test_data_dir}/B/03/0"
    iterator_configuration = IteratorConfiguration(roi_table="FOV_ROI_table")

    with pytest.raises(ValueError, match="None of the requested align_label_names"):
        compute_channel_registration_elastix(
            zarr_url=zarr_url,
            reference_channel=ChannelSelectionModel(
                mode="wavelength_id", identifier="A01_C01"
            ),
            calculate_on_labels=LabelInputModel(
                reference_label_name="nuclei",
                align_label_names=["nonexistent_label_1", "nonexistent_label_2"],
            ),
            parameter_files=parameter_files,
            iterator_configuration=iterator_configuration,
            level_path=0,
        )
