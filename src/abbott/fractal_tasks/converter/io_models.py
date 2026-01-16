# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

"""Io models for abbott H5 legacy to OME-Zarr converter Fractal task"""

from enum import Enum

from pydantic import BaseModel, Field


class ConverterOmeroChannel(BaseModel):
    """Custom class for Omero channels, based on OME-NGFF v0.4.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `405`.
        index: Do not change. For internal use only.
        label: Name of the channel.
        new_label: Optional new name for the channel.
    """

    # Custom

    wavelength_id: int
    index: int | None = None

    # From OME-NGFF v0.4 transitional metadata

    label: str | None = None
    new_label: str | None = None


class ConverterMultiplexingAcquisition(BaseModel):
    """Input class for Multiplexing Cellvoyager converter

    Attributes:
        allowed_image_channels: A list of `OmeroChannel` image objects, where each
            channel must include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
        allowed_label_channels: Optional list of `OmeroChannel` label objects, where
            each channel must include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
    """

    allowed_image_channels: list[ConverterOmeroChannel]
    allowed_label_channels: list[ConverterOmeroChannel] | None = None


class AllowedH5Extensions(str, Enum):
    """Enum for allowed H5 file extensions."""

    H5 = ".h5"
    HDF5 = ".hdf5"


class InitArgsCellVoyagerH5toOMEZarr(BaseModel):
    """Arguments to be passed from cellvoyager converter init to compute

    Attributes:
        input_files: List of (filtered) input H5 files to be converted.
        acquisitions: `MultiplexingAcquisition` object
            containing the channel information.
        well_ID: part of the image filename needed for finding the
            right subset of image files
        plate_path: Name of the plate, used to create the OME-Zarr path.
        mrf_path: Path to the MRF file for metadata extraction.
        mlf_path: Path to the MLF file for metadata extraction.
        include_glob_patterns: List of glob patterns to include files for
            metadata parsing.
        exclude_glob_patterns: List of glob patterns to exclude files for
            metadata parsing.
        overwrite: Whether to overwrite existing OME-Zarr data.
    """

    input_files: list[str]
    acquisition: ConverterMultiplexingAcquisition
    well_ID: str
    plate_path: str
    mrf_path: str
    mlf_path: str
    include_glob_patterns: list[str] | None = None
    exclude_glob_patterns: list[str] | None = None
    overwrite: bool


class ConverterWavelengthModel(BaseModel):
    """Input model for wavelength conversion.

    Attributes:
        wavelength_abbott_legacy: The name of the wavelength as set in the
            abbott legacy h5 file.
        wavelength_omezarr: Wavelength as used in OME-Zarr. Check in raw tif
            files for the correct wavelength name. E.g. "A01_C01" if DAPI channel
            was imaged first ("A01") and the first channel ("C01"). If green and
            far-red channel would be next imaged together, it would be "A02_C02"
            and "A02_C04" respectively.
    """

    wavelength_abbott_legacy: int
    wavelength_omezarr: str


def _default_wavelength():
    return [
        ConverterWavelengthModel(
            wavelength_abbott_legacy=405, wavelength_omezarr="A01_C01"
        ),
        ConverterWavelengthModel(
            wavelength_abbott_legacy=488, wavelength_omezarr="A02_C02"
        ),
        ConverterWavelengthModel(
            wavelength_abbott_legacy=561, wavelength_omezarr="A03_C03"
        ),
        ConverterWavelengthModel(
            wavelength_abbott_legacy=640, wavelength_omezarr="A04_C04"
        ),
    ]


class CustomWavelengthInputModel(BaseModel):
    """Input model for the custom wavelength conversions to be used in task.

    Attributes:
        wavelengths (list[WavelengthModel]): The list of wavelengths imaged in the
            abbott legacy h5 file and its corresponding OME-Zarr wavelengths following
            the OME-Zarr multiplexed naming conventions.
    """

    wavelengths: list[ConverterWavelengthModel] = Field(
        default_factory=_default_wavelength
    )


class ConverterOMEZarrBuilderParams(BaseModel):
    """Parameters for the OME-Zarr builder.

    Attributes:
        number_multiscale: The number of multiscale
            levels to create. Default is 4.
        xy_scaling_factor: The factor to downsample the XY plane.
            Default is 2, meaning every layer is half the size over XY.
        z_scaling_factor: The factor to downsample the Z plane.
            Default is 1, no scaling on Z.
        max_xy_chunk: The maximum size of the XY chunk.
        z_chunk: The size of the Z chunk.
        c_chunk: The size of the C chunk.

    """

    number_multiscale: int = Field(default=4, ge=0)
    xy_scaling_factor: int = Field(
        default=2,
        ge=1,
        le=10,
        title="Scaling Factor xy",
    )
    z_scaling_factor: int = Field(default=1, ge=1, le=10)
    max_xy_chunk: int = Field(default=4096, ge=1)
    z_chunk: int = Field(default=10, ge=1)
    c_chunk: int = Field(default=1, ge=1)
