import numpy as np

import torch

from ..primitives import inside_outside_function, points_to_cuboid_distances, \
    transform_to_primitives_centric_system, deform, \
    ray_plane_intersections

from ..utils.stats_logger import StatsLogger


def _euclidean_dual_loss_impl(
    X,
    prim_params,
    sampler,
    use_chamfer=True,
    use_cuboid=False
):
    """
    Arguments:
    ----------
        X: Tensor of size BxNx3 containing the points sampled from the surface
           of the target mesh
        prim_params: PrimitiveParameters object containing the predictions
                     of the network
    """
    # Get some sizes: batch size (B), number of points (N), (M) number of
    # primitives and S the number of points sampled from the SQs
    B, N, _ = X.shape
    M = prim_params.n_primitives
    S = sampler.n_samples

    probs = prim_params.probs
    translations = prim_params.translations_r
    rotations = prim_params.rotations_r
    alphas = prim_params.sizes_r
    epsilons = prim_params.shapes_r
    if prim_params.deformations is None:
        # Initialize with zero deformations
        tapering_params = probs.new_zeros(B, M, 2)
    else:
        tapering_params = prim_params.deformations_r


    # Transform the 3D points from world-coordinates to primitive-centric
    # coordinates with size BxNxMx3
    X_transformed = transform_to_primitives_centric_system(
        X,
        translations,
        rotations
    )
    # Get the coordinates of the sampled points on the surfaces of the SQs,
    # with size BxMxSx3
    X_SQ, normals = sampler.sample_points_on_primitive(use_cuboid, alphas, epsilons)
    X_SQ = deform(X_SQ, alphas, tapering_params)
    # Make the normals unit vectors
    normals_norm = normals.norm(dim=-1).view(B, M, S, 1)
    normals = normals / normals_norm

    # Make sure that everything has the right size
    assert X_SQ.shape == (B, M, S, 3)
    assert normals.shape == (B, M, S, 3)
    assert X_transformed.shape == (B, N, M, 3)
    # Make sure that the normals are unit vectors
    assert torch.sqrt(torch.sum(normals ** 2, -1)).sum() == B*M*S
    # Compute the pairwise Euclidean distances between points sampled on the
    # surface of the SQ (X_SQ) with points sampled on the surface of the target
    # object (X_transformed)
    V = (X_SQ.unsqueeze(3) - (X_transformed.permute(0, 2, 1, 3)).unsqueeze(2))
    assert V.shape == (B, M, S, N, 3)
    D = torch.sum((V)**2, -1)

    cvrg_loss, inside = euclidean_coverage_loss(
        [probs, translations, rotations, alphas, epsilons, tapering_params],
        X_transformed,
        D,
        use_cuboid,
        use_chamfer
    )
    assert inside is None or inside.shape == (B, N, M)

    cnst_loss = euclidean_consistency_loss(
        prim_params,
        V,
        normals,
        inside,
        D,
        use_chamfer
    )

    return cvrg_loss, cnst_loss, X_SQ


def euclidean_dual_loss( y_hat, y_target, sampler, options):
    """
    Arguments:
    ----------
        y_hat: PrimitiveParameters object containing the predictions
                     of the network
        y_target: Tensor with size BxNx6 with the N points from the target
                  object and their corresponding normals
        sampler: An object of either CuboidSampler or EqualDistanceSampler
                 depending on the type of the primitive we are using
        options: A dictionary with various options

    Returns:
    --------
        the loss
    """
    use_cuboid = options.get("use_cuboid", False)
    use_chamfer = options.get("use_chamfer", False)
    loss_weights = options["loss_weights"]

    gt_points = y_target[:, :, :3]
    # Make sure that everything has the right shape
    assert gt_points.shape[-1] == 3

    cvrg_loss, cnst_loss, X_SQ = _euclidean_dual_loss_impl(
        gt_points, y_hat, sampler
    )

    stats = StatsLogger.instance()
    stats["losses.cvrg"] = cvrg_loss.item()
    stats["losses.cnst"] = cnst_loss.item()
    stats["metrics.exp_n_prims"] = y_hat.probs.sum(-1).mean().item()

    w1 = loss_weights["coverage_loss_weight"]
    w2 = loss_weights["consistency_loss_weight"]
    return w1 * cvrg_loss + w2 * cnst_loss


def euclidean_coverage_loss(
    y_hat,
    X_transformed,
    D,
    use_cuboid=False,
    use_chamfer=False
):
    """
    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        X_transformed: Tensor with size BxNxMx3 with the N points from the
                       target object transformed in the M primitive-centric
                       coordinate systems
        D: Tensor of size BxMxSxN that contains the pairwise distances between
           points on the surface of the SQ to the points on the target object
    """
    # Declare some variables
    B = X_transformed.shape[0]  # batch size
    N = X_transformed.shape[1]  # number of points per sample
    M = X_transformed.shape[2]  # number of primitives

    shapes = y_hat[3].view(B, M, 3)
    epsilons = y_hat[4].view(B, M, 2)
    probs = y_hat[0]

    # Get the relative position of points with respect to the SQs using the
    # inside-outside function
    F = shapes.new_tensor(0)
    inside = None
    if not use_chamfer:
        if use_cuboid:
            F = points_to_cuboid_distances(X_transformed, shapes)
            inside = F <= 0
        else:
            F = inside_outside_function(
                X_transformed,
                shapes,
                epsilons
            )
            inside = F <= 1

    D = torch.min(D, 2)[0].permute(0, 2, 1)  # size BxNxM
    assert D.shape == (B, N, M)

    if not use_chamfer:
        D[inside] = 0.0
    distances, idxs = torch.sort(D, dim=-1)

    # Start by computing the cumulative product
    # Sort based on the indices
    probs = torch.cat([
        probs[i].take(idxs[i]).unsqueeze(0) for i in range(len(idxs))
    ])
    neg_cumprod = torch.cumprod(1-probs, dim=-1)
    neg_cumprod = torch.cat(
        [neg_cumprod.new_ones((B, N, 1)), neg_cumprod[:, :, :-1]],
        dim=-1
    )

    # minprob[i, j, k] is the probability that for sample i and point j the
    # k-th primitive has the minimum loss
    minprob = probs.mul(neg_cumprod)

    loss = torch.einsum("ijk,ijk->", [distances, minprob])
    loss = loss / B / N

    StatsLogger.instance()["F"] = F
    return loss, inside


def euclidean_consistency_loss(y_hat, V, normals, inside, D,
                               use_chamfer=False):
    """
    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        V: Tensor with size BxMxSxN3 with the vectors from the points on SQs to
           the points on the target's object surface.
        normals: Tensor with size BxMxSx3 with the normals at every sampled
                 points on the surfaces of the M primitives
        inside: A mask containing 1 if a point is inside the corresponding
                shape
        D: Tensor of size BxMxSxN that contains the pairwise distances between
           points on the surface of the SQ to the points on the target object
    """
    B = V.shape[0]  # batch size
    M = V.shape[1]  # number of primitives
    S = V.shape[2]  # number of points sampled on the SQ
    N = V.shape[3]  # number of points sampled on the target object
    probs = y_hat[0]

    assert D.shape == (B, M, S, N)

    # We need to compute the distance to the closest point from the target
    # object for every point S
    # min_D = D.min(-1)[0] # min_D has size BxMxS
    if not use_chamfer:
        outside = (1-inside).permute(0, 2, 1).unsqueeze(2).float()
        assert outside.shape == (B, M, 1, N)
        D = D + (outside*1e30)
    # Compute the minimum distances D, with size BxMxS
    D = D.min(-1)[0]
    D[D >= 1e30] = 0.0
    assert D.shape == (B, M, S)

    # Compute an approximate area of the superellipsoid as if it were an
    # ellipsoid
    shapes = y_hat[3].view(B, M, 3)
    area = 4 * np.pi * (
        (shapes[:, :, 0] * shapes[:, :, 1])**1.6 / 3 +
        (shapes[:, :, 0] * shapes[:, :, 2])**1.6 / 3 +
        (shapes[:, :, 1] * shapes[:, :, 2])**1.6 / 3
    )**0.625
    area = M * area / area.sum(dim=-1, keepdim=True)

    # loss = torch.einsum("ij,ij,ij->", [torch.max(D, -1)[0], probs, volumes])
    # loss = torch.einsum("ij,ij,ij->", [torch.mean(D, -1), probs, volumes])
    # loss = torch.einsum("ij,ij->", [torch.max(D, -1)[0], probs])
    loss = torch.einsum("ij,ij,ij->", [torch.mean(D, -1), probs, area])
    loss = loss / B / M

    return loss
