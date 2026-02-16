"""Pydantic models for advanced iterator configuration."""

from pydantic import BaseModel, Field


class IteratorConfiguration(BaseModel):
    """Advanced configuration.

    Attributes:
        roi_table (str | None): Name of a ROI table. If set, the segmentation
            will be applied to each ROI in the table individually. This option can
            be combined with masking.
    """

    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
