"""Deterministic CNN and DeiT input contracts from manuscript S1.

These are separate input representations. Converting a CNN IQ8 cache into the
historical DeiT PNG cache is not established by this module.
"""

import cv2
import numpy as np
import torch
from PIL import Image


def _gray_uint8(array):
    values = np.asarray(array)
    if values.ndim != 2 or not values.size:
        raise ValueError("Expected a nonempty two-dimensional grayscale ROI")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
        raise ValueError("Grayscale values must be finite integers in 0..255")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError("Expected the 8-bit integer representation, not continuous intensities")
    return values.astype(np.uint8)


def iq8_from_dicom_roi(pixels, slope=1.0, intercept=0.0):
    """The caller supplies an already aligned tight ROI from MONOCHROME2 DICOM."""
    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Expected an already cropped tight DICOM ROI")
    values = values * float(slope) + float(intercept)
    if not np.isfinite(values).all() or float(values.max()) <= float(values.min()):
        raise ValueError("Constant or nonfinite ROI is outside the frozen IQ8 contract")
    values = (values - values.min()) / (values.max() - values.min()) * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def cnn_iq8_tensor(iq8):
    image = _gray_uint8(iq8)
    height, width = image.shape
    scale = min(64.0 / height, 64.0 / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = 64 - new_h, 64 - new_w
    canvas = cv2.copyMakeBorder(resized, dh // 2, dh - dh // 2,
                               dw // 2, dw - dw // 2, cv2.BORDER_REPLICATE)
    values = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(values.copy()).unsqueeze(0)


def deit_png_tensor(png_roi):
    image = _gray_uint8(png_roi)
    image = cv2.equalizeHist(image)
    image = np.repeat(image[:, :, None], 3, axis=2)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    # Preserve the frozen Albumentations uint8 Normalize float32 LUT order.
    lut = (np.arange(256, dtype=np.float32) - np.float32(127.5)) * np.float32(1.0 / 127.5)
    values = cv2.LUT(image, lut)
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def reference_cached_tensor(source40_png):
    """Normalize the producer's already cropped/PIL-resized 224x224 gray cache."""
    image = _gray_uint8(source40_png)
    if image.shape != (224, 224):
        raise ValueError("Expected the reference producer's 224x224 source40 ROI")
    tensor = torch.from_numpy(image.copy()).float().div_(255.0)
    return ((tensor - 0.449) / 0.226).unsqueeze(0)


def reference_roi_tensor(full_png_image, bbox):
    """PIL grayscale -> exact 40%-each-side crop -> PIL bilinear -> .449/.226."""
    image = full_png_image.convert("L") if isinstance(full_png_image, Image.Image) else Image.fromarray(_gray_uint8(full_png_image))
    xmin, ymin, xmax, ymax = map(float, bbox)
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid reference bounding box")
    bw, bh = xmax - xmin, ymax - ymin
    width, height = image.size
    x0, x1 = max(0, int(xmin - .40*bw)), min(width, int(xmax + .40*bw))
    y0, y1 = max(0, int(ymin - .40*bh)), min(height, int(ymax + .40*bh))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Reference bounding box has no valid crop")
    crop = image.crop((x0,y0,x1,y1)).resize((224,224), Image.Resampling.BILINEAR)
    return reference_cached_tensor(np.asarray(crop, dtype=np.uint8))
