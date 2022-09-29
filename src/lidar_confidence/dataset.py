"""
Contains the utilities to manage the dataset
"""

import torch
from typing import Literal
from PIL import Image
import numpy as np
import os

__all__ = ["Dataset"]


def _identity(x):
    return x


class Dataset(torch.utils.data.Dataset):
    """
    Loads the data used for this project provinding different splits:

    train
        A subset composed by 8641 samples of KITTI Depth Completion
        (http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).
    val
        The whole validation split provided in KITTI Depth Completion.
    test_1
        Official 1000 samples used in KITTI Depth Completion to test.
    test_2
        142 samples extracted from KITTI Stereo 2015
        (http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

    Each sample contains the rgb image (320x1216x3), the lidar and groundtruth sparse
    data (320x1216x1)

    .. note::
        You can download the dataset using
        `dvc get https://github.com/andreaconti/lidar-confidence data/dataset`.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test_1", "test_2"] = "train",
        transform=_identity,
    ):
        assert split in ["train", "val", "test_1", "test_2"]

        # fields
        self.root = root
        self.split = split
        self.transform = transform

    def __len__(self):
        n_elem = len(os.listdir(os.path.join(self.root, self.split, "groundtruth")))
        return n_elem

    def __getitem__(self, x):
        out = {}
        idx = str(x).zfill(10)

        # load groundtruth, lidar and image
        out["gt"] = np.array(
            Image.open(os.path.join(self.root, self.split, "groundtruth", idx + ".png"))
        )[..., None]
        out["gt"] = out["gt"] / 256
        out["gt"] = out["gt"].astype(np.float32)

        out["lidar"] = np.array(
            Image.open(os.path.join(self.root, self.split, "lidar", idx + ".png"))
        )[..., None]
        out["lidar"] = out["lidar"] / 256
        out["lidar"] = out["lidar"].astype(np.float32)

        out["img"] = np.array(
            Image.open(os.path.join(self.root, self.split, "img", idx + ".jpg"))
        )

        return self.transform(out)
