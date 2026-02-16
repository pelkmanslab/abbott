"""Wrappers for working with Python ITK LabelImages and LabelMaps.

Be aware of the difference: A LabelImage is a regular itk.Image
of integer pixel type, a LabelMap is a special image data type storing LabelObjects.
"""

from typing import Optional, Union

import itk
import numpy as np

from abbott.registration.conversions import to_itk, to_labelmap, to_numpy


def median(img: itk.Image, radius: int):
    """Applies a median filter to an itk image.

    Attributes:
        img: itk.Image, the image to be filtered
        radius: int, the radius of the median filter
    """
    medianFilter = itk.MedianImageFilter[type(img), type(img)].New()
    medianFilter.SetRadius(radius)
    medianFilter.SetInput(img)
    medianFilter.Update()
    return medianFilter.GetOutput()


def pad_labelMap(
    label_map: itk.LabelMap, distance: int = 1, constant: int = 0
) -> itk.LabelMap:
    """Pads a LabelMap with a constant value.

    Attributes:
        label_map: itk.LabelMap, the LabelMap to be padded
        distance: int, the distance to pad the LabelMap
        constant: int, the constant value to pad with
    """
    label_img = to_itk(label_map)
    pad = itk.ConstantPadImageFilter.New(label_img)
    pad.SetConstant(constant)
    pad.SetPadBound(distance)
    pad.Update()
    return to_labelmap(pad.GetOutput())


def apply_image_filter(
    lbl_map: itk.LabelMap,
    filter_type: str,
    radius: Optional[int] = None,
    kernel: Union[np.ndarray, itk.FlatStructuringElement] = None,
    binary_internal_output=False,
    enable_filters_at_edge=True,
) -> itk.LabelMap:
    """Applies a morphological image filter to a LabelMap

    Attributes:
        lbl_map: itk.LabelMap, contains at least 1 object that is non zero
        filter_type: str, selection of erosion, dilation, opening & closing
        radius: int or tuple, optional, specifies the radius of the kernel to be used.
            Either radius or kernel needs to be provided
        kernel: a numpy or itk.FlatStructuringElement kernel. Optional, either radius
            or kernel needs to be provided
        binary_internal_output: Boolean, whether labels should be recalculated
                                after each operation. If set to True, objects
                                that are split will get separate labels.
        enable_filters_at_edge: Boolean, whether LabelMap should be padded such that
            filters also work for the pixels touching the image border
    """
    # Set the structuring element
    if radius is not None and kernel is not None:
        raise Exception(
            "Function is overspecified: Both a radius and a kernel were provided. "
            "Pick one."
        )
    elif radius is not None:
        StructuringElementType = itk.FlatStructuringElement[3]
        structuringElement = StructuringElementType.Ball(radius)
    elif kernel is not None:
        if type(kernel) == np.ndarray:
            # TODO: Find a way to convert a numpy array into an ITK kernel. Problem:
            # StructuringElementType.FromImage() takes a binary ITK image, but
            # binary ITK images don't appear to be wrapped for python.
            raise NotImplementedError
        else:
            structuringElement = kernel
        # Still define radius, because it's needed for padding
        np.max(structuringElement.GetRadius())
    else:
        raise Exception(
            "Function is underspecified. Neither radius nor kernel are provided. "
            "Provide one of them."
        )

    # Pad the label map by 1 such that filters also work at the edge of images
    if enable_filters_at_edge:
        lbl_map = pad_labelMap(lbl_map)

    # Pick the morphological filter
    image_filter = {
        "erosion": itk.BinaryErodeImageFilter[
            itk.Image[itk.UC, 3], itk.Image[itk.UC, 3], StructuringElementType
        ].New(),
        "dilation": itk.BinaryDilateImageFilter[
            itk.Image[itk.UC, 3], itk.Image[itk.UC, 3], StructuringElementType
        ].New(),
        "opening": itk.BinaryMorphologicalOpeningImageFilter[
            itk.Image[itk.UC, 3], itk.Image[itk.UC, 3], StructuringElementType
        ].New(),
        "closing": itk.BinaryMorphologicalClosingImageFilter[
            itk.Image[itk.UC, 3], itk.Image[itk.UC, 3], StructuringElementType
        ].New(),
    }[filter_type]

    # Apply the filter
    image_filter.SetKernel(structuringElement)
    objectByObjectLabelMapFilter = itk.ObjectByObjectLabelMapFilter.LM3.New()
    if filter_type in ["dilation", "closing"]:
        objectByObjectLabelMapFilter.SetPadSize(radius)
    objectByObjectLabelMapFilter.SetInput(lbl_map)
    objectByObjectLabelMapFilter.SetBinaryInternalOutput(binary_internal_output)
    objectByObjectLabelMapFilter.SetFilter(image_filter)

    UniqueLabelMapFilterType = itk.LabelUniqueLabelMapFilter.LM3
    unique = UniqueLabelMapFilterType.New()
    unique.SetInput(objectByObjectLabelMapFilter.GetOutput())
    unique.Update()

    if enable_filters_at_edge:
        # Can this be done directly in ITK? Do I loose important info with this,
        # besides the manually reset spacing?
        output_img = unique.GetOutput()
        output_spacing = tuple(output_img.GetSpacing())
        return to_labelmap(to_numpy(output_img)[1:-1, 1:-1, 1:-1], scale=output_spacing)
    else:
        return unique.GetOutput()
