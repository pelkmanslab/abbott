"""Helper functions for elastix registration tasks.

Some are moodified from
https://github.com/fractal-analytics-platform/fractal-cellpose-sam-task/blob/main/src/fractal_cellpose_sam_task/utils.py
"""

import logging
from typing import Literal

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IteratorConfiguration(BaseModel):
    """Advanced configuration.

    Attributes:
        roi_table (str | None): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")


class ElastixReferenceChannel(BaseModel):
    """Elastix reference channel configuration.

    This model is used to select a reference channel by label, wavelength ID, or index.

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    """
    Specifies how to interpret the identifiers. Can be "label", "wavelength_id", or
    "index" (must be an integer).
    """
    identifiers: list[str] = Field(min_length=1, max_length=1)
    """
    Unique identifiers for the channels. This can be channel labels, wavelength IDs, or
    indices, depending on the mode.
    At least one and at most three identifiers must be provided.
    """

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]


class LabelInputModel(BaseModel):
    """Configuration for label-based channel registration.

    When provided, the registration transform is calculated on label images
    instead of intensity images.

    Attributes:
        reference_label_name: Name of the label image to use as reference.
        align_label_names: Names of label images to align against the
            reference. Must contain at least one entry.
    """

    reference_label_name: str
    align_label_names: list[str]
