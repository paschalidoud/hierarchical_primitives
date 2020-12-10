
import torch

from ..networks.primitive_parameters import PrimitiveParameters
from ..primitives import sq_volumes


def always(*args):
    return True


def qos_less(qos_th):
    """Split iff qos is less than qos_th."""
    def inner(P, depth, idx):
        return P[depth].qos[0, idx] < qos_th
    return inner


def volume_larger(vol_th):
    """Accepts iff the volume is larger that vol_th."""
    def inner(P, depth, idx):
        return sq_volumes(P[depth])[0, idx] > vol_th
    return inner


def filter_primitives(P, predicate_split, predicate_accept):
    """Compute the indices of the leaves selected based on the provided
    predicates for splitting and accepting a primitive."""
    # P should only contain one primitive
    for p in P:
        assert len(p.sizes) == 1

    # Do a depth first search with filters for split and accept
    primitives = []
    nodes = [(0, 0)]
    max_depth = len(P)-1
    while nodes:
        depth, idx = nodes.pop()
        if depth == max_depth or not predicate_split(P, depth, idx):
            if predicate_accept(P, depth, idx):
                primitives.append((depth, idx))
        else:
            nodes.append((depth+1, 2*idx))
            nodes.append((depth+1, 2*idx+1))

    return primitives


def primitive_parameters_from_indices(P, indices):
    B = 1
    M = len(indices)
    return PrimitiveParameters.with_keys(
        probs=torch.ones(B, M),
        translations=torch.stack([
            P[depth].translations_r[:, idx]
            for depth, idx in indices
        ], dim=1).view(B, -1),
        rotations=torch.stack([
            P[depth].rotations_r[:, idx]
            for depth, idx in indices
        ], dim=1).view(B, -1),
        sizes=torch.stack([
            P[depth].sizes_r[:, idx]
            for depth, idx in indices
        ], dim=1).view(B, -1),
        shapes=torch.stack([
            P[depth].shapes_r[:, idx]
            for depth, idx in indices
        ], dim=1).view(B, -1),
        sharpness=torch.stack([
            P[depth].shapes_r[:, idx]
            for depth, idx in indices
        ], dim=1).view(B, -1)
    )


def get_primitives_indices(P):
    """Compute the indices of the leaves"""
    # P should only contain one primitive
    for p in P:
        assert len(p.sizes) == 1

    # Do a depth first search with filters for split and accept
    primitives = []
    nodes = [(0, 0)]
    max_depth = len(P)-1
    while nodes:
        depth, idx = nodes.pop()
        if depth == max_depth:
            primitives.append((depth, idx))
        else:
            nodes.append((depth+1, 2*idx))
            nodes.append((depth+1, 2*idx+1))

    return primitives
