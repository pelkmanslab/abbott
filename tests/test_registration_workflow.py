import shutil
from pathlib import Path

import pytest
import zarr
from devtools import debug

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
from abbott.registration.utils import IteratorConfiguration


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
    level = 2
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
        reference_wavelength=reference_wavelength,
        parameter_files=parameter_files,
        iterator_configuration=iterator_configuration,
        level_path=level,
        lower_rescale_quantile=0.0,
        upper_rescale_quantile=0.99,
    )

    # Test zarr_url that needs to be registered
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_wavelength=reference_wavelength,
        iterator_configuration=iterator_configuration,
        level_path=0,
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_url}_channels_registered"
    zarr.open_group(new_zarr_url, mode="r")

    # Pre-existing output can be overwritten
    apply_channel_registration_elastix(
        zarr_url=zarr_url,
        reference_wavelength=reference_wavelength,
        iterator_configuration=iterator_configuration,
        level_path=0,
        overwrite_input=True,
    )
