from pathlib import Path

import h5py
import numpy as np
import pytest

from abbott.fractal_tasks.convert_abbottlegacyh5_to_omezarr_compute import (
    convert_abbottlegacyh5_to_omezarr_compute,
)
from abbott.fractal_tasks.convert_abbottlegacyh5_to_omezarr_init import (
    convert_abbottlegacyh5_to_omezarr_init,
)
from abbott.fractal_tasks.converter.io_models import (
    AllowedH5Extensions,
    CustomWavelengthInputModel,
    MultiplexingAcquisition,
    OMEZarrBuilderParams,
)


def create_h5(
    f: h5py.File,
    dset_name: str,
    data: np.array,
    stain: str,
    cycle: int,
    wavelength: int,
    level: int = 0,
    scale: tuple[float, float, float] = (1, 0.2, 0.2),
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
def sample_h5_file_3d(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a sample h5 file for testing."""
    # tmp_path = Path("/data/active/rhornb/pytest/conversion/")
    tmp_path.mkdir(parents=True, exist_ok=True)
    h5_file = tmp_path / "B02_px-1229_py-0112.h5"
    random_image_c0 = np.random.randint(0, 255, (10, 10, 10))
    random_image_c1 = np.random.randint(0, 255, (10, 10, 10))
    random_label_c0 = np.random.randint(0, 2, (10, 10, 10))
    with h5py.File(h5_file, "w") as f:
        create_h5(
            f,
            dset_name="ch_00/0",
            data=random_image_c0,
            stain="DAPI",
            cycle=0,
            wavelength=405,
        )
        create_h5(
            f,
            dset_name="ch_01/0",
            data=random_image_c1,
            stain="FITC",
            cycle=1,
            wavelength=488,
        )
        create_h5(
            f,
            dset_name="nuclei/0",
            data=random_label_c0,
            stain="nuclei",
            cycle=0,
            wavelength=405,
            img_type="label",
        )
    return h5_file


def test_full_workflow_3D(sample_h5_file_3d: tuple[str, str], tmp_path: Path):
    # tmp_path = Path("/data/active/rhornb/pytest/conversion/")
    zarr_dir = str(tmp_path / "test.zarr")
    image_path = sample_h5_file_3d
    image_path = str(image_path)

    allowed_image_channels_c0 = [
        {
            "wavelength_id": 405,
            "label": "DAPI",
            "new_label": "DAPI_0",
        },
    ]
    allowed_image_channels_c1 = [
        {
            "wavelength_id": 488,
            "label": "FITC",
        },
    ]
    allowed_label_channels_c0 = [
        {
            "wavelength_id": 405,
            "label": "nuclei",
        },
    ]

    acquisitions = {
        "0": MultiplexingAcquisition(
            allowed_image_channels=allowed_image_channels_c0,
            allowed_label_channels=allowed_label_channels_c0,
        ),
        "1": MultiplexingAcquisition(
            allowed_image_channels=allowed_image_channels_c1,
        ),
    }

    ome_zarr_parameters = OMEZarrBuilderParams(
        number_multiscale=2,
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
        input_dir=str(tmp_path),
        acquisitions=acquisitions,
        include_glob_patterns=None,
        exclude_glob_patterns=None,
        h5_extension=AllowedH5Extensions.H5,
        mrf_path=str(Path(__file__).parent / "data/MeasurementDetail.mrf"),
        mlf_path=str(Path(__file__).parent / "data/MeasurementData.mlf"),
    )["parallelization_list"]

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label="nuclei",
            overwrite=True,
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        assert Path(zarr_url).exists()
