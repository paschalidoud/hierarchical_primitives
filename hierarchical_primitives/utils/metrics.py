import numpy as np
import torch


def compute_iou(occ1, occ2, weights=None, average=True):
    """Compute the intersection over union (IoU) for two sets of occupancy
    values.

    Arguments:
    ----------
        occ1: Tensor of size BxN containing the first set of occupancy values
        occ2: Tensor of size BxN containing the first set of occupancy values

    Returns:
    -------
        the IoU
    """
    if not torch.is_tensor(occ1):
        occ1 = torch.tensor(occ1)
        occ2 = torch.tensor(occ2)

    if weights is None:
        weights = occ1.new_ones(occ1.shape)

    assert len(occ1.shape) == 2
    assert occ1.shape == occ2.shape

    # Convert them to boolean
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IoU
    area_union = (occ1 | occ2).float()
    area_union = (weights * area_union).sum(dim=-1)
    area_union = torch.max(area_union.new_tensor(1.0), area_union)
    area_intersect = (occ1 & occ2).float()
    area_intersect = (weights * area_intersect).sum(dim=-1)
    iou = (area_intersect / area_union)

    if average:
        return iou.mean().item()
    else:
        return iou
