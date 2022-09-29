"""
Custom metrics used in this project
"""
from typing import Literal, Tuple
import torch

__all__ = [
    "percentiles",
    "cost_curve_aucs",
    "cost_curve_auc",
    "cost_curve_optimal_auc",
    "cost_curve_random_auc",
]


def percentiles(x: torch.Tensor, percentiles: torch.Tensor) -> torch.Tensor:
    """
    Compute the percentiles of input

    Parameters
    ----------
    x: torch.Tensor
        Input tensor
    percentiles: 1D torch.Tensor
        Contains the percentiles to be computed, for instance [25, 50, 75]

    Returns
    -------
    percentiles: Nx1 torch.Tensor
        Returns the percentiles computed

    Examples
    --------
    >>> percentiles(torch.arange(300), [25, 50, 75])
    tensor([[ 74],
            [149],
            [224]])
    """
    if not isinstance(percentiles, torch.Tensor):
        percentiles = torch.tensor(percentiles, dtype=x.dtype, device=x.device)

    x = x.view(x.shape[0], -1)
    in_sorted, in_argsort = torch.sort(x, dim=0)
    positions = percentiles * (x.shape[0] - 1) / 100
    floored = torch.floor(positions)
    ceiled = floored + 1
    ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
    weight_ceiled = positions - floored
    weight_floored = 1.0 - weight_ceiled
    d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
    d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
    result = (d0 + d1).view(-1, *x.shape[1:])
    return result.type(x.dtype)


def cost_curve_aucs(
    groundtruth: torch.Tensor,
    sparse_depth: torch.Tensor,
    confidences: torch.Tensor,
    criterion: Literal["mae", "rmse"] = "mae",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    auc = cost_curve_auc(groundtruth, sparse_depth, confidences, criterion)
    auc_opt = cost_curve_optimal_auc(groundtruth, sparse_depth, criterion)
    auc_rnd = cost_curve_random_auc(groundtruth, sparse_depth, criterion)

    return auc, auc_opt, auc_rnd


def cost_curve_auc(
    groundtruth: torch.Tensor,
    sparse_depth: torch.Tensor,
    confidences: torch.Tensor,
    criterion: Literal["mae", "rmse"] = "mae",
) -> torch.Tensor:
    """
    Computes a cost curve sorting measures by means of :attr:`confs` and incrementally
    computing the error obtained between :attr:`groundtruth` and
    :attr:`sparse_depth` using the chosen :attr:`criterion`.

    Finally it computes the area under such curve: lower is better.

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
    out: torch.Tensor
        the area under the cost curve
    """
    assert criterion in ["mae", "rmse"], "criterion mae or rmse"

    AUCs = []
    quants = torch.tensor(
        [2 * t for t in range(1, 50)],
        device=groundtruth.device,
        dtype=groundtruth.dtype,
    )

    for gt, depth, conf in zip(groundtruth, sparse_depth, confidences):
        gt, depth, conf = gt[0], depth[0], conf[0]

        valid = (gt > 0) & (depth > 0)
        cmap = conf[valid]
        gt = gt[valid]
        depth = depth[valid]

        ROC = []
        cthresholds = percentiles(cmap, quants).flatten()

        csubs = [cmap >= t for t in cthresholds]

        if criterion == "mae":
            ROC.append(torch.abs(depth - gt).mean())
            ROC_points = [(torch.abs(depth - gt))[s].mean() for s in csubs]
        else:
            ROC.append(torch.sqrt(torch.mean(torch.square(depth - gt))))
            ROC_points = [
                torch.sqrt(torch.mean(torch.square(depth - gt)[s])) for s in csubs
            ]

        [ROC.append(r) for r in ROC_points]
        ROC.append(0)

        AUC = torch.trapz(torch.tensor(ROC, device=gt.device), dx=1.0 / 50.0)
        AUCs.append(AUC)

    return torch.mean(
        torch.tensor(AUCs, dtype=groundtruth.dtype, device=groundtruth.device)
    )


def cost_curve_optimal_auc(
    groundtruth: torch.Tensor,
    sparse_depth: torch.Tensor,
    criterion: Literal["mae", "rmse"] = "mae",
) -> torch.Tensor:
    """
    Computes a cost curve sorting measures in the best way and incrementally
    computing the error obtained between :attr:`groundtruth` and
    :attr:`sparse_depth` using the chosen :attr:`criterion`.

    Finally it computes the area under such curve: lower is better, this
    function returns the best reachable AUC.

    Parameters
    ----------
    groundtruth: Bx1xHxW torch.Tensor
        The groundtruth depth map where 0 is regarded as missing value
    sparse_depth: Bx1xHxW torch.Tensor
        The sparse depth map where 0 is regarded as missing value
    criterion: mae or rmse, default mae
        the criterion used to compute the loss

    Returns
    -------
    out: torch.Tensor
        the area under the cost curve
    """
    assert criterion in ["mae", "rmse"], "criterion mae or rmse"

    optAUCs = []
    quants = torch.tensor(
        [2 * t for t in range(1, 50)],
        device=groundtruth.device,
        dtype=groundtruth.dtype,
    )

    for gt, depth in zip(groundtruth, sparse_depth):
        gt, depth = gt[0], depth[0]

        # get valid points (where both lidar and gt are defined)
        valid = (gt > 0) & (depth > 0)

        # keep valid pixels only
        gt, depth = gt[valid], depth[valid]

        optROC = []

        # get thresholds according to confidence or to error map
        emap = -torch.abs(depth - gt) if criterion == "mae" else -((depth - gt) ** 2)
        ethresholds = percentiles(emap, quants)
        esubs = [emap >= t for t in ethresholds]

        # first point in the curve: total error
        if criterion == "mae":
            optROC.append(torch.abs(depth - gt).mean())
            optROC_points = [(torch.abs(depth - gt))[s].mean() for s in esubs]
        else:
            optROC.append(torch.sqrt(torch.mean(torch.square(depth - gt))))
            optROC_points = [
                torch.sqrt(torch.mean(torch.square(depth - gt)[s])) for s in esubs
            ]

        [optROC.append(r) for r in optROC_points]
        optROC.append(0)

        optAUC = torch.trapz(torch.tensor(optROC, device=gt.device), dx=1.0 / 50.0)
        optAUCs.append(optAUC)

    return torch.tensor(optAUCs, dtype=gt.dtype, device=gt.device).mean()


def cost_curve_random_auc(
    groundtruth: torch.Tensor,
    sparse_depth: torch.Tensor,
    criterion: Literal["mae", "rmse"] = "mae",
) -> torch.Tensor:
    """
    Computes a cost curve without any sorting or processing and incrementally
    computing the error obtained between :attr:`groundtruth` and
    :attr:`sparse_depth` using the chosen :attr:`criterion`.

    Finally it computes the area under such curve: lower is better, this value is
    similar to the auc obtained with a random confidence generator.

    Parameters
    ----------
    groundtruth: Bx1xHxW torch.Tensor
        The groundtruth depth map where 0 is regarded as missing value
    sparse_depth: Bx1xHxW torch.Tensor
        The sparse depth map where 0 is regarded as missing value
    criterion: mae or rmse, default mae
        the criterion used to compute the loss

    Returns
    -------
    out: torch.Tensor
        the area under the cost curve
    """
    assert criterion in ["mae", "rmse"], "criterion mae or rmse"
    rndAUCs = []

    for gt, depth in zip(groundtruth, sparse_depth):
        gt, depth = gt[0], depth[0]

        valid = (gt > 0) & (depth > 0)
        gt = gt[valid]
        depth = depth[valid]

        if criterion == "mae":
            rndAUCs.append(torch.abs(depth - gt).mean())
        else:
            rndAUCs.append(torch.sqrt(torch.mean(torch.square(depth - gt))))

    return torch.tensor(rndAUCs, device=gt.device).mean()
