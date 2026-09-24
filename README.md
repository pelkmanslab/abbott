# abbott
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI Status](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml)
[![codecov](https://codecov.io/github/pelkmanslab/abbott/graph/badge.svg?token=BF9NP4YLO6)](https://codecov.io/github/pelkmanslab/abbott)

A [Fractal](https://fractal-analytics-platform.github.io/) task collection for 3D multiplexed image analysis: registration of multiplexed cycles and channels, plus helper tasks for conversion and label processing.

## Available Tasks

### Registration

| Task | Description |
| --- | --- |
| Compute Registration (elastix) | Computes rigid/affine/b-spline registration to align multiplexed 3D images across cycles. |
| Apply Registration (elastix) | Applies the computed elastix registration to images. |
| Compute Registration (warpfield) | Computes [warpfield](https://github.com/danionella/warpfield) registration to align multiplexed 3D images across cycles. Requires a GPU. |
| Apply Registration (warpfield) | Applies the computed warpfield registration. Requires a GPU. |
| Compute Channel Registration (elastix) | Computes similarity registration of all channels in an acquisition to a reference channel. |
| Apply Channel Registration (elastix) | Applies channel registration to a multi-channel acquisition. |

> [!IMPORTANT]
> The warpfield registration tasks currently require CUDA > 11.x.

### Conversion

| Task | Description |
| --- | --- |
| Convert abbott-legacy H5 to OME-Zarr | Converts H5 files in the abbott-legacy format to OME-Zarr. |

### Image Processing

| Task | Description |
| --- | --- |
| Upsample Label Image | Upsamples label images to the highest image resolution. Useful when segmentation was performed on a lower resolution level (e.g. level 1), to avoid resolution mismatches in downstream tasks. |

An example multiplexing workflow is available in [examples/](examples/).

### Moved or discontinued tasks

| Task | Status |
| --- | --- |
| Convert Cellvoyager Multiplexing to existing OME-Zarr | **Discontinued.** Use `Convert Yokogawa CellVoyager Plate to OME-Zarr` from [fractal-uzh-converters](https://github.com/fractal-analytics-platform/fractal-uzh-converters). |
| Stardist Segmentation | Moved to [abbott-segmentation-tasks](https://github.com/pelkmanslab/abbott-segmentation-tasks). |
| Seeded Watershed Segmentation | Moved to [abbott-segmentation-tasks](https://github.com/pelkmanslab/abbott-segmentation-tasks). |

For feature extraction, see [abbott-features](https://github.com/pelkmanslab/abbott-features).

## Installation

### On a Fractal server

Download the `.tar.gz` from the latest [GitHub release](https://github.com/pelkmanslab/abbott/releases) and install it with Fractal's pixi task collection.

### Locally

Requires Python 3.11.

```bash
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e .
```

## Development

The development environment is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/pelkmanslab/abbott
cd abbott
pixi run init-tasks    # install pre-commit hooks, format code, build manifest, run tests
```

Individual tasks:

```bash
pixi run -e dev create-manifest   # regenerate __FRACTAL_MANIFEST__.json
pixi run -e dev format-code       # ruff format
pixi run -e test test             # run the test suite
```
