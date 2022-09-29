"""
Utilities to handle images
"""

import numpy as np
import torch
from typing import Union
import warnings
import cmapy


__all__ = ["normalize", "invert_depth", "color_depth"]


_T = Union[torch.Tensor, np.ndarray]


def normalize(img: _T, min: int = 0, max: int = 1) -> _T:
    """
    Normalizes a tensor or numpy array inside a specific range (min, max)
    """
    img_min, img_max = img.min(), img.max()
    normalized = min + (img - img_min) * (max - min) / (img_max - img_min)
    return normalized


def invert_depth(depth: _T, *, missing_value=None) -> _T:
    """
    Computes the reciprocal of :attr:``depth`` filtering out :attr:``missing_value``
    values found in the array. It doesn't deal with zero division (if you don't mask
    zeros it will lead to an inf.
    """

    # ignore erroneous divide by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv_depth = 1 / depth

    # mask missing values
    if missing_value is not None:
        inv_depth[depth == missing_value] = missing_value

    return inv_depth


def color_depth(depth: _T, *, invert=False, cmap: str = "viridis"):
    """
    Colorize a depth map with a colormap.

    Parameters
    ----------
    depth: np.ndarray HxWx1 or torch.Tensor 1xHxW
        the depth map.
    invert: bool, default False
        if invert the depth map before being colored

    Returns
    -------
    out: np.ndarray HxWx1 or torch.Tensor 3xHxW
        the same type of depth with dtype uint8
    """

    if invert:
        depth = invert_depth(depth, missing_value=0)
    depth = normalize(depth, 0, 255)

    depth_ = depth
    if isinstance(depth, torch.Tensor):
        depth_ = depth.permute(1, 2, 0).cpu().numpy()
    depth_ = depth_.astype(np.uint8)

    colored_image = cmapy.colorize(depth_, cmap, rgb_order=True)

    if isinstance(depth, torch.Tensor):
        colored_image = (
            torch.from_numpy(colored_image).to(depth.device).permute(2, 0, 1)
        )
    return colored_image
