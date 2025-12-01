# abbott
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI Status](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml)
[![codecov](https://codecov.io/github/pelkmanslab/abbott/graph/badge.svg?token=BF9NP4YLO6)](https://codecov.io/github/pelkmanslab/abbott)

3D Multiplexed Image Analysis Task Collection

## Available Tasks

| Task | Description | Passing |
| --- | --- | --- |
| Convert Cellvoyager Multiplexing to existing OME-Zarr | Converts CV7000/CV8000 images and extends to existing OME-Zarr file.| ✓ |
| Convert Abbott-legacy H5 to OME-Zarr | Converts H5 files in abbott-legacy format to OME-Zarr files| ✓ |
| Compute Registration (elastix) | Compute rigid/affine/b-spline registration for aligning multiplexed 3D images across cycles.|✓|
| Apply Registration (elastix) | Apply rigid/affine/b-spline registration to images.|✓|
| Compute Registration (warpfield) | Compute warpfield registration (see https://github.com/danionella/warpfield) for aligning multiplexed 3D images across cycles.|✓|
| Apply Registration (warpfield) | Apply warpfield registration.|✓|
| Compute Channel Registration (elastix) | Compute similarity registration of all channels in an acquisition to a reference channel.|✓|
| Apply Channel Registration (elastix) | Apply similarity registration to multi-channel acquisition.|✓|
| Stardist Segmentation | Segment images using Stardist. |✓|
| Seeded Watershed Segmentation | Performs segmentation (e.g., of cells) using a label image as seeds and an intensity image (e.g., membrane stain) for boundary detection. |✓|
| Upsample Label Image | Upsamples label images to the highest image resolution. Useful if segmentation was peformed on e.g. level 1 to avoid resolution mismatch in downstream tasks. |✓|

> [!IMPORTANT]
> Warpfield Registration Tasks currently only run with CUDA > 12.x. See task documentation for instructions on how to run the tasks on the Pelkmans cluster.

## Installation

To install this task package on a Fractal server, get the whl from the Github release and use the local task collection.

To install this package locally:
```
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e .
```
To also run Stardist task locally:
```
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e ".[stardist]"
```

For development:
```
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e ".[dev,stardist]" 
```
