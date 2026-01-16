from pathlib import Path

import h5py
import numpy as np
import pytest
from ngio import open_ome_zarr_container

from abbott.fractal_tasks.convert_abbottlegacyh5_to_omezarr_compute import (
    convert_abbottlegacyh5_to_omezarr_compute,
)
from abbott.fractal_tasks.convert_abbottlegacyh5_to_omezarr_init import (
    convert_abbottlegacyh5_to_omezarr_init,
)
from abbott.fractal_tasks.converter.io_models import (
    AllowedH5Extensions,
    ConverterMultiplexingAcquisition,
    ConverterOMEZarrBuilderParams,
    CustomWavelengthInputModel,
)


def create_h5(
    f: h5py.File,
    dset_name: str,
    data: np.array,
    stain: str,
    cycle: int,
    wavelength: int,
    level: int = 0,
    scale: tuple[float, float, float] = (1.0, 0.322, 0.322),
    img_type: str = "intensity",
):
    """Create a dataset in an HDF5 file."""

    f.create_dataset(
        dset_name,
        data=data,
        compression="gzip",
        chunks=True,
    )
    f[dset_name].attrs["element_size_um"] = np.array(scale, dtype=np.float64)
    f[dset_name].attrs["img_type"] = img_type
    f[dset_name].attrs["stain"] = stain
    f[dset_name].attrs["cycle"] = cycle
    f[dset_name].attrs["wavelength"] = wavelength
    f[dset_name].attrs["level"] = level


@pytest.fixture
def sample_h5_file_3d(tmp_path: Path):
    """Create a sample h5 file for testing."""
    tmp_path = Path(tmp_path) / "data"
    tmp_path.mkdir(parents=True, exist_ok=True)
    h5_file_path_1 = tmp_path / "B02_px-1849_py-0958.h5"
    h5_file_path_2 = tmp_path / "B03_px+0101_py+0728.h5"
    # h5_file_path_2 = tmp_path / "B02_px-2514_py+0114.h5"
    random_image = np.random.randint(0, 3, (15, 2000, 2000))
    random_label_c0_file_1 = np.zeros((15, 2000, 2000), dtype=np.int32)
    random_label_c0_file_1[:, 100:900, 100:900] = 1
    random_label_c0_file_1[:, 1100:1900, 1100:1900] = 2
    random_label_c0_file_2 = np.zeros((15, 2000, 2000), dtype=np.int32)
    random_label_c0_file_2[:, 100:900, 100:900] = 3

    with h5py.File(h5_file_path_1, "w") as f:
        create_h5(
            f,
            dset_name="ch_00/0",
            data=random_image,
            stain="DAPI",
            cycle=0,
            wavelength=405,
        )
        create_h5(
            f,
            dset_name="ch_01/0",
            data=random_image,
            stain="FITC",
            cycle=0,
            wavelength=405,
        )
        create_h5(
            f,
            dset_name="ch_02/0",
            data=random_image,
            stain="FITC",
            cycle=1,
            wavelength=488,
        )
        create_h5(
            f,
            dset_name="nuclei/0",
            data=random_label_c0_file_1,
            stain="nuclei",
            cycle=0,
            wavelength=405,
            img_type="label",
        )

    with h5py.File(h5_file_path_2, "w") as f:
        create_h5(
            f,
            dset_name="ch_00/0",
            data=random_image,
            stain="DAPI",
            cycle=0,
            wavelength=405,
        )
        create_h5(
            f,
            dset_name="ch_01/0",
            data=random_image,
            stain="FITC",
            cycle=0,
            wavelength=405,
        )
        create_h5(
            f,
            dset_name="ch_02/0",
            data=random_image,
            stain="FITC",
            cycle=1,
            wavelength=488,
        )
        create_h5(
            f,
            dset_name="nuclei/0",
            data=random_label_c0_file_2,
            stain="nuclei",
            cycle=0,
            wavelength=405,
            img_type="label",
        )
    return [h5_file_path_1, h5_file_path_2]


def test_full_workflow_3D(sample_h5_file_3d: list[Path], tmp_path: Path):
    zarr_dir = tmp_path.as_posix()
    allowed_image_channels_c0 = [
        {
            "wavelength_id": 405,
            "label": "DAPI",
            "new_label": "DAPI_0",
        },
        {
            "wavelength_id": 488,
            "label": "FITC",
            "new_label": "FITC_0",
        },
    ]
    allowed_image_channels_c1 = [
        {
            "wavelength_id": 488,
            "label": "FITC",
            "new_label": "FITC_1",
        },
    ]
    allowed_label_channels_c0 = [
        {
            "wavelength_id": 405,
            "label": "nuclei",
        },
    ]

    acquisitions = {
        "0": ConverterMultiplexingAcquisition(
            allowed_image_channels=allowed_image_channels_c0,
            allowed_label_channels=allowed_label_channels_c0,
        ),
        "1": ConverterMultiplexingAcquisition(
            allowed_image_channels=allowed_image_channels_c1,
        ),
    }

    ome_zarr_parameters = ConverterOMEZarrBuilderParams(
        number_multiscale=4,
        xy_scaling_factor=2,
        z_scaling_factor=1,
        create_all_ome_axis=True,
    )

    wavelengths = CustomWavelengthInputModel(
        wavelengths=[
            {"wavelength_abbott_legacy": 405, "wavelength_omezarr": "A01_C01"},
            {"wavelength_abbott_legacy": 488, "wavelength_omezarr": "A02_C02"},
        ]
    )

    parallelization_list = convert_abbottlegacyh5_to_omezarr_init(
        zarr_dir=zarr_dir,
        input_dir=sample_h5_file_3d[0].parent.as_posix(),
        acquisitions=acquisitions,
        include_glob_patterns=["*B03*"],
        exclude_glob_patterns=None,
        h5_extension=AllowedH5Extensions.H5,
        mrf_path=str(Path(__file__).parent / "data/MeasurementDetail.mrf"),
        mlf_path=str(Path(__file__).parent / "data/MeasurementData.mlf"),
        overwrite=False,
    )["parallelization_list"]

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label="nuclei",
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        assert Path(zarr_url).exists()
        # Make sure no errors are raised when opening the OME Zarr container and table
        ome_zarr_container = open_ome_zarr_container(zarr_url)
        ome_zarr_container.get_table("FOV_ROI_table")
