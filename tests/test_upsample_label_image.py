import logging
import shutil
from pathlib import Path

import dask.array as da
import fractal_tasks_core
import numpy as np
import pytest
import zarr
from devtools import debug
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.utils import rescale_datasets
from ngio.utils._errors import NgioValueError

from abbott.fractal_tasks.upsample_label_image import upsample_label_image

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def test_data_dir_3d(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "upsampled_3d").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_upsample_label_image_same_resolutions(test_data_dir_3d):
    zarr_urls = [f"{test_data_dir_3d}/B/03/0", f"{test_data_dir_3d}/B/03/1"]

    upsample_label_image(
        zarr_url=zarr_urls[0],
        label_name="emb_linked",
        # output_label_name="emb_linked_upsampled",
        output_ROI_table="emb_ROI_table_2_linked",
        overwrite=True,
    )


def test_upsample_label_image_lower_resolution(test_data_dir_3d):
    zarr_urls = [f"{test_data_dir_3d}/B/03/0", f"{test_data_dir_3d}/B/03/1"]
    label_name = "label_upsample"

    zarr_url = zarr_urls[0]
    label_zarr_url = Path(f"{zarr_url}/labels/emb_linked")
    label_image = da.from_zarr(f"{label_zarr_url}/1").compute()

    # Update zarr group
    ngff_image_meta = load_NgffImageMeta(str(zarr_url))
    coarsening_xy = ngff_image_meta.coarsening_xy
    num_levels = ngff_image_meta.num_levels
    new_datasets = rescale_datasets(
        datasets=[ds.model_dump() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=1,
        remove_channel_axis=True,
    )
    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.model_dump()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    # Prepare new output label path
    image_group = zarr.group(zarr_url)
    prepare_label_group(
        image_group,
        label_name,
        overwrite=True,
        label_attrs=label_attrs,
        logger=logger,
    )

    label_path_out = f"{zarr_url}/labels/{label_name}/0"
    store = zarr.storage.FSStore(label_path_out)
    label_dtype = np.uint32

    # Ensure that all output shapes & chunks are 3D (for 2D data: (1, y, x))
    # https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/398
    data_zyx = da.from_zarr(f"{zarr_url}/{1}")[0]
    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    da.array(label_image).to_zarr(
        url=mask_zarr,
        compute=True,
    )

    # Create pyramid
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{label_name}",
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )

    output_label_name = "label_upsampled"
    upsample_label_image(
        zarr_url=zarr_url,
        label_name=label_name,
        output_label_name=output_label_name,
        output_ROI_table="emb_ROI_table_upsampled",
        overwrite=True,
    )

    # Check that the upsampled label image has the same shape as the
    # original (correct) label image
    label_zarr_url = Path(f"{zarr_url}/labels/{output_label_name}")
    label_image_new = da.from_zarr(f"{label_zarr_url}/1").compute()
    assert label_image.shape == label_image_new.shape

    # Test FileNotFoundError if label_name does not exist
    with pytest.raises(NgioValueError):
        upsample_label_image(
            zarr_url=zarr_urls[1],
            label_name=label_name,
            output_label_name=output_label_name,
            output_ROI_table="emb_ROI_table_upsampled",
            overwrite=True,
        )
