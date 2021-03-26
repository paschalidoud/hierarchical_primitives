from itertools import combinations
import os

import numpy as np

from hierarchical_primitives.common.base import build_dataloader
from hierarchical_primitives.networks.base import build_network
from hierarchical_primitives.primitives import quaternions_to_rotation_matrices
from hierarchical_primitives.utils.visualization_utils import points_on_sq_surface

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def is_inside(pcl1, pcl2, threshold):
    # Check the percentage of points that lie inside another pointcloud and if
    # they exceed a threashold return True, else return False
    assert pcl1.shape[0] == 3
    assert pcl2.shape[0] == 3
    # for every point in pcl2 check whether it lies inside pcl1
    minimum = pcl1.min(1)
    maximum = pcl1.max(1)

    c1 = np.logical_and(
        pcl2[0, :] <= maximum[0],
        pcl2[0, :] >= minimum[0],
    )
    c2 = np.logical_and(
        pcl2[1, :] <= maximum[1],
        pcl2[1, :] >= minimum[1],
    )
    c3 = np.logical_and(
        pcl2[2, :] <= maximum[2],
        pcl2[2, :] >= minimum[2],
    )
    c4 = np.logical_and(c1, np.logical_and(c2, c3)).sum()
    return float(c4) / pcl1.shape[1] > threshold


def get_non_overlapping_primitives(y_hat, active_prims, insidness=0.6):
    n_primitives = y_hat.n_primitives
    points_from_prims = []

    R = quaternions_to_rotation_matrices(
            y_hat.rotations.view(-1, 4)
    ).to("cpu").detach()
    translations = y_hat.translations.to("cpu").view(-1, 3)
    translations = translations.detach().numpy()

    shapes = y_hat.sizes.view(-1, 3).detach().numpy()
    epsilons = y_hat.shapes.to("cpu").view(-1, 2).detach().numpy()
    taperings = np.zeros((n_primitives, 2))

    prim_pts = []
    for i in active_prims:
            x_tr, y_tr, z_tr, prim_pts =\
                points_on_sq_surface(
                    shapes[i, 0],
                    shapes[i, 1],
                    shapes[i, 2],
                    epsilons[i, 0],
                    epsilons[i, 1],
                    R[i].numpy(),
                    translations[i].reshape(-1, 1),
                    taperings[i, 0],
                    taperings[i, 1]
                )
            points_from_prims.append(prim_pts)

    cmp1 = combinations(active_prims, 2)
    cmp2 = combinations(points_from_prims, 2)
    non_overlapping_prims = active_prims[:]
    for (i, j), (pcl1, pcl2) in zip(cmp1, cmp2):
        if is_inside(pcl1, pcl2, insidness) and j in non_overlapping_prims:
            non_overlapping_prims.remove(j)
    return non_overlapping_prims


def build_dataloader_and_network_from_args(args, config, device="cpu"):
    # Create a dataloader instance to generate the samples for training
    dataloader = build_dataloader(
        config,
        args.model_tags,
        args.category_tags,
        split=["train", "test", "val"],
        batch_size=1,
        n_processes=4,
        random_subset=args.random_subset,
    )

    # Build the network architecture to be used for training
    network = build_network(args.config_file, args.weight_file, device=device)
    network.eval()

    return dataloader, network
