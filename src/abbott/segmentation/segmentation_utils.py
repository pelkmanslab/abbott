# Copyright 2023 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Joel Lüthi  <joel.luethi@fmi.ch>
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Helper functions to run segmentation tasks"""

import logging
from typing import Literal, Optional

import numpy as np
from fractal_tasks_core.tasks.cellpose_utils import normalized_img
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class SeededSegmentationCustomNormalizer(BaseModel):
    """Validator to handle different normalization scenarios for seeded

    segmentation task.


    If `norm_type="no_normalization"`, then no normalization is used and no
    other parameters can be specified.
    If `norm_type="custom"`, then either percentiles or explicit integer
    bounds can be applied.

    Attributes:
        norm_type:
            One of type `custom`(using the custom parameters) or `no_normalization`.
        lower_percentile: Specify a custom lower-bound percentile for rescaling
            as a float value between 0 and 100. You can only specify percentiles
            or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for rescaling
            as a float value between 0 and 100.
            You can only specify percentiles or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentiles or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000.
            You can only specify percentiles or bounds, not both.
    """

    norm_type: Literal["custom", "no_normalization"] = "no_normalization"
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    # In the future, add an option to allow using precomputed percentiles
    # that are stored in OME-Zarr histograms and use this pydantic model that
    # those histograms actually exist

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        # Extract values
        norm_type = self.norm_type
        lower_percentile = self.lower_percentile
        upper_percentile = self.upper_percentile
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Verify that custom parameters are only provided when type="custom"
        if norm_type != "custom":
            if lower_percentile is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {lower_percentile=}. "
                    "Hint: set type='custom'."
                )
            if upper_percentile is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {upper_percentile=}. "
                    "Hint: set norm_type='custom'."
                )
            if lower_bound is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {lower_bound=}. "
                    "Hint: set norm_type='custom'."
                )
            if upper_bound is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {upper_bound=}. "
                    "Hint: set norm_type='custom'."
                )

        # The only valid options are:
        # 1. Both percentiles are set and both bounds are unset
        # 2. Both bounds are set and both percentiles are unset
        are_percentiles_set = (
            lower_percentile is not None,
            upper_percentile is not None,
        )
        are_bounds_set = (
            lower_bound is not None,
            upper_bound is not None,
        )
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set " "together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError("Both lower_bound and upper_bound must be set together")
        if lower_percentile is not None and lower_bound is not None:
            raise ValueError(
                "You cannot set both explicit bounds and percentile bounds "
                "at the same time. Hint: use only one of the two options."
            )

        return self


def normalize_seeded_segmentation_channel(
    x: np.ndarray,
    normalization: SeededSegmentationCustomNormalizer,
) -> np.ndarray:
    """Normalize a seeded segmentation input array by channel.

    Args:
        x: 3D numpy array.
        normalization: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.

    """
    # Optionally perform custom normalization
    if normalization.norm_type == "custom":
        x = normalized_img(
            x,
            lower_p=normalization.lower_percentile,
            upper_p=normalization.upper_percentile,
            lower_bound=normalization.lower_bound,
            upper_bound=normalization.upper_bound,
        )

    return x
