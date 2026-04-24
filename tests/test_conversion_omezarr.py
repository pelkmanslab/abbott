from pathlib import Path

import h5py
import numpy as np
import pytest
from fractal_tasks_core.cellvoyager.metadata import parse_yokogawa_metadata
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
from abbott.fractal_tasks.converter.task_utils import find_inconsistent_z_field_patterns

MRF_PATH = str(Path(__file__).parent / "data/MeasurementDetail.mrf")
MLF_PATH = str(Path(__file__).parent / "data/MeasurementData.mlf")


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


def _write_h5_files(
    base_dir: Path,
    level: int,
    xy_size: int,
    scale: tuple[float, float, float],
    include_labels: bool = True,
) -> list[Path]:
    """Write two H5 test files (B02 and B03) with the given level and scale."""
    h5_file_path_1 = base_dir / "B02_px-1849_py-0958.h5"
    h5_file_path_2 = base_dir / "B03_px+0101_py+0728.h5"
    lvl = str(level)

    random_image = np.random.randint(0, 3, (15, xy_size, xy_size), dtype=np.int32)
    random_label_1 = np.zeros((15, xy_size, xy_size), dtype=np.int32)
    random_label_1[
        :, xy_size // 20 : xy_size * 9 // 20, xy_size // 20 : xy_size * 9 // 20
    ] = 1
    random_label_1[
        :,
        xy_size * 11 // 20 : xy_size * 19 // 20,
        xy_size * 11 // 20 : xy_size * 19 // 20,
    ] = 2
    random_label_2 = np.zeros((15, xy_size, xy_size), dtype=np.int32)
    random_label_2[
        :, xy_size // 20 : xy_size * 9 // 20, xy_size // 20 : xy_size * 9 // 20
    ] = 3

    for path, label_data in [
        (h5_file_path_1, random_label_1),
        (h5_file_path_2, random_label_2),
    ]:
        with h5py.File(path, "w") as f:
            create_h5(f, f"ch_00/{lvl}", random_image, "DAPI", 0, 405, level, scale)
            create_h5(f, f"ch_01/{lvl}", random_image, "FITC", 0, 405, level, scale)
            create_h5(f, f"ch_02/{lvl}", random_image, "FITC", 1, 488, level, scale)
            if include_labels:
                create_h5(
                    f,
                    f"nuclei/{lvl}",
                    label_data,
                    "nuclei",
                    0,
                    405,
                    level,
                    scale,
                    img_type="label",
                )

    return [h5_file_path_1, h5_file_path_2]


@pytest.fixture
def sample_h5_file_3d(tmp_path: Path):
    """Full-resolution (level=0) H5 files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return _write_h5_files(data_dir, level=0, xy_size=2000, scale=(1.0, 0.322, 0.322))


@pytest.fixture
def sample_h5_file_3d_level1(tmp_path: Path):
    """Downsampled (level=1) H5 files with doubled XY pixel size."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return _write_h5_files(data_dir, level=1, xy_size=1000, scale=(1.0, 0.644, 0.644))


@pytest.fixture
def sample_h5_file_3d_no_labels(tmp_path: Path):
    """Full-resolution H5 files without any label datasets."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return _write_h5_files(
        data_dir, level=0, xy_size=2000, scale=(1.0, 0.322, 0.322), include_labels=False
    )


@pytest.fixture
def common_params():
    """Shared acquisitions, wavelengths, and OME-Zarr builder parameters."""
    acquisitions = {
        "0": ConverterMultiplexingAcquisition(
            allowed_image_channels=[
                {"wavelength_id": 405, "label": "DAPI", "new_label": "DAPI_0"},
                {"wavelength_id": 488, "label": "FITC", "new_label": "FITC_0"},
            ],
            allowed_label_channels=[
                {"wavelength_id": 405, "label": "nuclei"},
            ],
        ),
        "1": ConverterMultiplexingAcquisition(
            allowed_image_channels=[
                {"wavelength_id": 488, "label": "FITC", "new_label": "FITC_1"},
            ],
        ),
    }
    wavelengths = CustomWavelengthInputModel(
        wavelengths=[
            {"wavelength_abbott_legacy": 405, "wavelength_omezarr": "A01_C01"},
            {"wavelength_abbott_legacy": 488, "wavelength_omezarr": "A02_C02"},
        ]
    )
    ome_zarr_parameters = ConverterOMEZarrBuilderParams(
        number_multiscale=4,
        xy_scaling_factor=2,
        z_scaling_factor=1,
        create_all_ome_axis=True,
    )
    return acquisitions, wavelengths, ome_zarr_parameters


def _run_init(zarr_dir, input_dir, acquisitions, include_glob_patterns=None):
    return convert_abbottlegacyh5_to_omezarr_init(
        zarr_dir=zarr_dir,
        input_dir=input_dir,
        acquisitions=acquisitions,
        include_glob_patterns=include_glob_patterns,
        exclude_glob_patterns=None,
        h5_extension=AllowedH5Extensions.H5,
        mrf_path=MRF_PATH,
        mlf_path=MLF_PATH,
        overwrite=False,
    )["parallelization_list"]


def test_full_workflow_3D(sample_h5_file_3d, tmp_path, common_params):
    """Full-resolution workflow: check channels, labels, tables, pixel size,
    and metadata."""
    acquisitions, wavelengths, ome_zarr_parameters = common_params
    zarr_dir = tmp_path.as_posix()
    input_dir = sample_h5_file_3d[0].parent.as_posix()

    parallelization_list = _run_init(
        zarr_dir, input_dir, acquisitions, include_glob_patterns=["*B03*"]
    )

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label="nuclei",
        )

        update = image_list_update["image_list_updates"][0]
        zarr_url = update["zarr_url"]
        acq_id = update["attributes"]["acquisition"]

        # Return structure
        assert Path(zarr_url).exists()
        assert update["types"]["is_3D"] is True
        assert update["attributes"]["well"] == "B03"

        container = open_ome_zarr_container(zarr_url)
        tables = container.list_tables()

        # Tables present for all acquisitions
        assert "FOV_ROI_table" in tables
        assert "well_ROI_table" in tables

        # Pixel size comes from the H5 scale attribute
        pixel_size = container.get_image().pixel_size
        assert pytest.approx(pixel_size.x, rel=1e-3) == 0.322
        assert pytest.approx(pixel_size.y, rel=1e-3) == 0.322
        assert pytest.approx(pixel_size.z, rel=1e-3) == 1.0

        if acq_id == "0":
            assert container.channel_labels == ["DAPI_0", "FITC_0"]
            assert container.num_channels == 2
            assert "nuclei" in container.list_labels()
            assert "nuclei_ROI_table" in tables
        else:
            assert acq_id == "1"
            assert container.channel_labels == ["FITC_1"]
            assert container.num_channels == 1
            assert container.list_labels() == []
            assert "nuclei_ROI_table" not in tables


def test_full_workflow_3D_level1(sample_h5_file_3d_level1, tmp_path, common_params):
    """Downsampled (level=1) workflow: pixel size should reflect the level-1 scale."""
    acquisitions, wavelengths, ome_zarr_parameters = common_params
    zarr_dir = tmp_path.as_posix()
    input_dir = sample_h5_file_3d_level1[0].parent.as_posix()

    parallelization_list = _run_init(
        zarr_dir, input_dir, acquisitions, include_glob_patterns=["*B03*"]
    )

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=1,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label="nuclei",
        )

        update = image_list_update["image_list_updates"][0]
        zarr_url = update["zarr_url"]

        assert Path(zarr_url).exists()
        assert update["types"]["is_3D"] is True

        container = open_ome_zarr_container(zarr_url)
        assert "FOV_ROI_table" in container.list_tables()

        # Pixel size must reflect level-1 scale (0.644 µm), not level-0 (0.322 µm)
        pixel_size = container.get_image().pixel_size
        assert pytest.approx(pixel_size.x, rel=1e-3) == 0.644
        assert pytest.approx(pixel_size.y, rel=1e-3) == 0.644
        assert pytest.approx(pixel_size.z, rel=1e-3) == 1.0


def test_full_workflow_no_masking_label(sample_h5_file_3d, tmp_path, common_params):
    """Without masking_label: nuclei label is written but no masking ROI table
    is created."""
    acquisitions, wavelengths, ome_zarr_parameters = common_params
    zarr_dir = tmp_path.as_posix()
    input_dir = sample_h5_file_3d[0].parent.as_posix()

    parallelization_list = _run_init(
        zarr_dir, input_dir, acquisitions, include_glob_patterns=["*B03*"]
    )

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label=None,
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        container = open_ome_zarr_container(zarr_url)
        tables = container.list_tables()

        assert "nuclei_ROI_table" not in tables
        assert "FOV_ROI_table" in tables
        # Label itself is still written even without masking
        acq_id = image_list_update["image_list_updates"][0]["attributes"]["acquisition"]
        if acq_id == "0":
            assert "nuclei" in container.list_labels()


def test_full_workflow_no_label_channels(
    sample_h5_file_3d_no_labels, tmp_path, common_params
):
    """Acquisition with no label channels: no labels written to the OME-Zarr."""
    _, wavelengths, ome_zarr_parameters = common_params
    acquisitions_no_labels = {
        "0": ConverterMultiplexingAcquisition(
            allowed_image_channels=[
                {"wavelength_id": 405, "label": "DAPI", "new_label": "DAPI_0"},
                {"wavelength_id": 488, "label": "FITC", "new_label": "FITC_0"},
            ],
            allowed_label_channels=None,
        ),
    }

    zarr_dir = tmp_path.as_posix()
    input_dir = sample_h5_file_3d_no_labels[0].parent.as_posix()

    parallelization_list = _run_init(
        zarr_dir, input_dir, acquisitions_no_labels, include_glob_patterns=["*B03*"]
    )

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            masking_label=None,
        )

        zarr_url = image_list_update["image_list_updates"][0]["zarr_url"]
        container = open_ome_zarr_container(zarr_url)
        assert container.list_labels() == []
        assert container.channel_labels == ["DAPI_0", "FITC_0"]


def test_invalid_axes_raises(sample_h5_file_3d, tmp_path, common_params):
    """axes_names other than 'ZYX' must raise ValueError."""
    acquisitions, wavelengths, ome_zarr_parameters = common_params

    parallelization_list = _run_init(
        tmp_path.as_posix(),
        sample_h5_file_3d[0].parent.as_posix(),
        acquisitions,
        include_glob_patterns=["*B03*"],
    )

    with pytest.raises(ValueError, match="Unsupported axes names"):
        convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=parallelization_list[0]["zarr_url"],
            init_args=parallelization_list[0]["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
            axes_names="XYZ",
        )


def test_missing_level_raises(sample_h5_file_3d, tmp_path, common_params):
    """Requesting a level absent from the H5 file must raise FileNotFoundError."""
    acquisitions, wavelengths, ome_zarr_parameters = common_params

    parallelization_list = _run_init(
        tmp_path.as_posix(),
        sample_h5_file_3d[0].parent.as_posix(),
        acquisitions,
        include_glob_patterns=["*B03*"],
    )

    with pytest.raises(FileNotFoundError):
        convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=parallelization_list[0]["zarr_url"],
            init_args=parallelization_list[0]["init_args"],
            level=5,  # only level=0 exists in the fixture
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
        )


# ---------------------------------------------------------------------------
# Inconsistent Z steps
# ---------------------------------------------------------------------------


def _mlf_records(
    row: int, col: int, field: int, ch: int, n_z: int, x: float, y: float
) -> str:
    """Generate n_z IMG MeasurementRecord entries for one (well, field, channel)."""
    well = f"{chr(64 + row)}{col:02d}"
    lines = []
    for z_idx in range(1, n_z + 1):
        z = -50.0 + (z_idx - 1)
        fname = (
            f"AssayPlate_Greiner_CELLSTAR655090_{well}_"
            f"T0001F{field:03d}L01A01Z{z_idx:02d}C{ch:02d}.tif"
        )
        lines.append(
            f'<bts:MeasurementRecord bts:Type="IMG" '
            f'bts:Time="2024-04-08T16:00:00.000+02:00" '
            f'bts:Column="{col}" bts:Row="{row}" bts:TimePoint="1" '
            f'bts:FieldIndex="{field}" bts:ZIndex="{z_idx}" '
            f'bts:TimelineIndex="1" bts:ActionIndex="1" bts:Action="3D" '
            f'bts:X="{x}" bts:Y="{y}" bts:Z="{z:.1f}" '
            f'bts:Ch="{ch}">{fname}</bts:MeasurementRecord>'
        )
    return "\n".join(lines)


@pytest.fixture
def mlf_inconsistent_z(tmp_path: Path) -> str:
    """Minimal MLF where B03/F1 has mismatched Z counts across channels.

    B03/F1 Ch1 = 15 Z planes, Ch2 = 20 Z planes  →  inconsistent (triggers the check).
    B03/F3 Ch1 = 15 Z planes, Ch2 = 15 Z planes  →  consistent, X/Y match
    the test H5 file ``B03_px+0101_py+0728.h5`` (abs-rounded: x=101, y=728).
    """
    content = "\n".join(
        [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<bts:MeasurementData bts:Version="1.0" '
            'xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">',
            # B03 / F1 — INCONSISTENT: Ch1 has 15 Z, Ch2 has 20 Z
            _mlf_records(row=2, col=3, field=1, ch=1, n_z=15, x=-73.3, y=-1715.7),
            _mlf_records(row=2, col=3, field=1, ch=2, n_z=20, x=-73.3, y=-1715.7),
            # B03 / F3 — consistent, coordinates match the test H5 file
            _mlf_records(row=2, col=3, field=3, ch=1, n_z=15, x=101.0, y=-727.6),
            _mlf_records(row=2, col=3, field=3, ch=2, n_z=15, x=101.0, y=-727.6),
            "</bts:MeasurementData>",
        ]
    )
    out = tmp_path / "MeasurementData_inconsistent.mlf"
    out.write_text(content)
    return str(out)


def test_find_inconsistent_z_field_patterns(mlf_inconsistent_z):
    """Helper identifies the one field whose channel Z counts disagree."""
    patterns = find_inconsistent_z_field_patterns(
        mrf_path=MRF_PATH,
        mlf_path=mlf_inconsistent_z,
        include_patterns=["*B03*"],
    )
    assert patterns == ["*_B03_*F001L*"]


def test_workflow_with_inconsistent_z_mlf(
    sample_h5_file_3d, tmp_path, common_params, mlf_inconsistent_z
):
    """Workflow auto-excludes the inconsistent field and completes successfully.

    B03/F1 has mismatched Z counts → parse_yokogawa_metadata raises.
    The compute task detects the bad field, retries without it, and converts
    B03 using the valid F3 metadata (which matches the test H5 coordinates).
    """
    acquisitions, wavelengths, ome_zarr_parameters = common_params
    zarr_dir = tmp_path.as_posix()
    input_dir = sample_h5_file_3d[0].parent.as_posix()

    # Confirm the raw MLF does trigger the consistency error
    with pytest.raises(ValueError, match="consistency check failed"):
        parse_yokogawa_metadata(
            mrf_path=MRF_PATH,
            mlf_path=mlf_inconsistent_z,
            include_patterns=["*B03*"],
        )

    parallelization_list = convert_abbottlegacyh5_to_omezarr_init(
        zarr_dir=zarr_dir,
        input_dir=input_dir,
        acquisitions=acquisitions,
        include_glob_patterns=["*B03*"],
        exclude_glob_patterns=None,
        h5_extension=AllowedH5Extensions.H5,
        mrf_path=MRF_PATH,
        mlf_path=mlf_inconsistent_z,
        overwrite=False,
    )["parallelization_list"]

    for image in parallelization_list:
        image_list_update = convert_abbottlegacyh5_to_omezarr_compute(
            zarr_url=image["zarr_url"],
            init_args=image["init_args"],
            level=0,
            wavelengths=wavelengths,
            ome_zarr_parameters=ome_zarr_parameters,
        )
        update = image_list_update["image_list_updates"][0]
        assert Path(update["zarr_url"]).exists()
        assert update["types"]["is_3D"] is True
