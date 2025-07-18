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

from fractal_tasks_core.channels import Window
from pydantic import BaseModel, Field, field_validator


class OmeroChannel(BaseModel):
    """Custom class for Omero channels, based on OME-NGFF v0.4.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `405`.
        index: Do not change. For internal use only.
        label: Name of the channel.
        new_label: Optional new name for the channel.
        window: Optional `Window` object to set default display settings. If
            unset, it will be set to the full bit range of the image
            (e.g. 0-65535 for 16 bit images).
        color: Optional hex colormap to display the channel in napari (it
            must be of length 6, e.g. `00FFFF`).
        active: Should this channel be shown in the viewer?
        coefficient: Do not change. Omero-channel attribute.
        inverted: Do not change. Omero-channel attribute.
    """

    # Custom

    wavelength_id: int
    index: int | None = None

    # From OME-NGFF v0.4 transitional metadata

    label: str | None = None
    new_label: str | None = None
    window: Window | None = None
    color: str | None = None
    active: bool = True
    coefficient: int = 1
    inverted: bool = False

    @field_validator("color", mode="after")
    @classmethod
    def valid_hex_color(cls, v: str | None) -> str | None:
        """Check that `color` is made of exactly six elements which are letters

        (a-f or A-F) or digits (0-9).
        """
        if v is None:
            return v
        if len(v) != 6:
            raise ValueError(f'color must have length 6 (given: "{v}")')
        allowed_characters = "abcdefABCDEF0123456789"
        for character in v:
            if character not in allowed_characters:
                raise ValueError(
                    "color must only include characters from "
                    f'"{allowed_characters}" (given: "{v}")'
                )
        return v


class MultiplexingAcquisition(BaseModel):
    """Input class for Multiplexing Cellvoyager converter

    Attributes:
        allowed_image_channels: A list of `OmeroChannel` image objects, where each
            channel must include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
        allowed_label_channels: Optional list of `OmeroChannel` label objects, where
            each channel must include the `wavelength_id` attribute and where the
            `wavelength_id` values must be unique across the list.
    """

    allowed_image_channels: list[OmeroChannel]
    allowed_label_channels: list[OmeroChannel] | None = None


class AllowedH5Extensions(str, Enum):
    """Enum for allowed H5 file extensions."""

    H5 = ".h5"
    HDF5 = ".hdf5"


class InitArgsCellVoyagerH5toOMEZarr(BaseModel):
    """Arguments to be passed from cellvoyager converter init to compute

    Attributes:
        input_files: List of (filtered) input H5 files to be converted.
        acquisitions: Dictionary of `MultiplexingAcquisition` objects
            containing the cycle and channel information.
        well_ID: part of the image filename needed for finding the
            right subset of image files
        mrf_path: Path to the MRF file for metadata extraction.
        mlf_path: Path to the MLF file for metadata extraction.
    """

    input_files: list[str]
    acquisitions: dict[str, MultiplexingAcquisition]
    well_ID: str
    mrf_path: str
    mlf_path: str


class WavelengthModel(BaseModel):
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
        WavelengthModel(wavelength_abbott_legacy=405, wavelength_omezarr="A01_C01"),
        WavelengthModel(wavelength_abbott_legacy=488, wavelength_omezarr="A02_C02"),
        WavelengthModel(wavelength_abbott_legacy=568, wavelength_omezarr="A03_C03"),
        WavelengthModel(wavelength_abbott_legacy=647, wavelength_omezarr="A04_C04"),
    ]


class CustomWavelengthInputModel(BaseModel):
    """Input model for the custom wavelength conversions to be used in task.

    Attributes:
        wavelengths (list[WavelengthModel]): The list of wavelengths imaged in the
            abbott legacy h5 file and its corresponding OME-Zarr wavelengths following
            the OME-Zarr multiplexed naming conventions.
    """

    wavelengths: list[WavelengthModel] = Field(default_factory=_default_wavelength)


class OMEZarrBuilderParams(BaseModel):
    """Parameters for the OME-Zarr builder.

    Attributes:
        number_multiscale: The number of multiscale
            levels to create. Default is 4.
        xy_scaling_factor: The factor to downsample the XY plane.
            Default is 2, meaning every layer is half the size over XY.
        z_scaling_factor: The factor to downsample the Z plane.
            Default is 1, no scaling on Z.
        create_all_ome_axis: Whether to create all OME axis.
            Default is True, meaning that missing axis will be created
            with a singleton dimension.
    """

    number_multiscale: int = Field(default=4, ge=0)
    xy_scaling_factor: int = Field(
        default=2,
        ge=1,
        le=10,
        title="Scaling Factor xy",
    )
    z_scaling_factor: int = Field(default=1, ge=1, le=10)
