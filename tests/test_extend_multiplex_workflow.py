import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from devtools import debug
from fractal_tasks_core.tasks.cellvoyager_to_ome_zarr_init_multiplex import (
    cellvoyager_to_ome_zarr_init_multiplex,
)
from fractal_tasks_core.tasks.io_models import MultiplexingAcquisition

from abbott.fractal_tasks.cellvoyager_compute_omezarr import (
    cellvoyager_to_ome_zarr_compute,
)
from abbott.fractal_tasks.cellvoyager_to_ome_zarr_init_extend_multiplex import (
    cellvoyager_to_ome_zarr_init_extend_multiplex,
)

single_cycle_allowed_channels_no_label = [
    {
        "wavelength_id": "A01_C01",
        "color": "00FFFF",
        "window": {"start": 0, "end": 700},
    },
]

num_levels = 5
coarsening_xy = 2


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_images_multiplex: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "registration_data").as_posix()
    debug(zenodo_images_multiplex, dest_dir)
    shutil.copytree(zenodo_images_multiplex, dest_dir)
    return dest_dir


def test_extend_multiplexing_yokogawa_to_existing_ome_zarr(
    tmp_path: Path,
    zenodo_images_multiplex: Sequence[str],
):
    zarr_urls_init_extend = []

    acquisition_init = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[0],
            allowed_channels=single_cycle_allowed_channels_no_label,
        )
    }

    # Init fake first cycle
    zarr_dir = str(tmp_path / "tmp_out/")

    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_dir=zarr_dir,
        acquisitions=acquisition_init,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=None,
        overwrite=True,
    )["parallelization_list"]

    # Convert to OME-Zarr
    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )
        zarr_urls_init_extend.append(image["zarr_url"])

    #####
    # Extend zarr file with new cycle
    #####

    acquisition_extend = {
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[1],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
    }

    parallelization_list = cellvoyager_to_ome_zarr_init_extend_multiplex(
        zarr_dir=zarr_dir,
        acquisitions=acquisition_extend,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=None,
    )["parallelization_list"]

    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )
        zarr_urls_init_extend.append(image["zarr_url"])

    # Test adding cycle that has already been added

    parallelization_list = cellvoyager_to_ome_zarr_init_extend_multiplex(
        zarr_dir=zarr_dir,
        acquisitions=acquisition_extend,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=None,
        overwrite=True,
    )["parallelization_list"]

    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )

    #####
    # Assert if extended OME-Zarr file has the same amount of
    # zarr_urls as OME-Zarr files initiated in a single step
    #####

    zarr_urls_init = []
    acquisitions = {
        "0": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[0],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
        "1": MultiplexingAcquisition(
            image_dir=zenodo_images_multiplex[1],
            allowed_channels=single_cycle_allowed_channels_no_label,
        ),
    }

    zarr_dir_comb = str(tmp_path / "tmp_comb/")

    parallelization_list = cellvoyager_to_ome_zarr_init_multiplex(
        zarr_dir=zarr_dir_comb,
        acquisitions=acquisitions,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        image_extension="png",
        metadata_table_files=None,
        overwrite=True,
    )["parallelization_list"]

    # Convert to OME-Zarr
    for image in parallelization_list:
        cellvoyager_to_ome_zarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
        )
        zarr_urls_init.append(image["zarr_url"])

    assert len(zarr_urls_init) == len(zarr_urls_init_extend)
