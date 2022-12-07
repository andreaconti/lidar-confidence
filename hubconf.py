import pathlib as _pathlib
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import Literal as _Literal
from typing import Tuple as _Tuple

import torch as _torch
import torch.nn.functional as _F
from torch.utils.data import Dataset as _Dataset

from src.lidar_confidence.metrics import cost_curve_aucs as _aucs

# dependencies

dependencies = ["torch", "torchvision"]

# utilities to download the model with torchhub

_CURR_DIR = _pathlib.Path(__file__).parent
_CURR_VERS = "0.1.0"


def _get_archive(archive_name) -> _pathlib.Path:
    # create archives folder
    download_dir = _CURR_DIR / "hubdownload"
    download_dir.mkdir(parents=True, exist_ok=True)

    # download
    if not (download_dir / archive_name).exists():
        _torch.hub.download_url_to_file(
            f"https://github.com/andreaconti/lidar-confidence/releases/download/v{_CURR_VERS}/{archive_name}",
            str(download_dir / archive_name),
        )
    return download_dir / archive_name


# API


def model(dataset: str = "kitti"):
    """
    Download the pretrained model over a subset of KITTI Depth Completion
    """
    assert dataset == "kitti", "only kitti is available"
    model_path = _get_archive("model.pth")
    return _torch.jit.load(model_path)


def _identity(x):
    return x


def dataset_kitti_142(transform: _Callable[[_Dict], _Dict] = _identity) -> _Dataset:
    """
    Returns a Dataset instance initialized to load the KITTI 142 split
    """
    # imports
    import tarfile

    from src.lidar_confidence.dataset import Dataset

    # download and untar
    archive_path = _get_archive("kitti_142_split.tar")
    dataset_root = archive_path.parent / "dataset/test_2"
    if not dataset_root.exists():
        dataset_root.parent.mkdir(exist_ok=True, parents=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(dataset_root)

    # build dataset
    return Dataset(root=dataset_root.parent, split="test_2", transform=transform)


def auc() -> _Callable[
    [_torch.Tensor, _torch.Tensor, _torch.Tensor, _Literal["mae", "rmse"]],
    _Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor],
]:
    """
    Utility function to compute the :func:`cost_curve_auc`,
    :func:`cost_curve_optimal_auc` and :func:`cost_curve_random_auc`
    outputs in a single shot.

    Parameters
    ----------
    groundtruth: Bx1xHxW torch.Tensor
        The groundtruth depth map where 0 is regarded as missing value
    sparse_depth: Bx1xHxW torch.Tensor
        The sparse depth map where 0 is regarded as missing value
    confidences: Bx1xHxW torch.Tensor
        A map containing in each position where :attr:`groundtruth` > 0 and
        :attr:`sparse_depth` > 0 the confidence of that value. Is accepted
        as confidence value any float >= 0.
    criterion: mae or rmse, default mae
        the criterion used to compute the loss

    Returns
    -------
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        auc, optimal auc, random auc
    """
    return _aucs
