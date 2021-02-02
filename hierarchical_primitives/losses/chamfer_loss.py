import torch

from ..primitives import deform, transform_to_world_coordinates_system, \
    transform_to_primitives_centric_system, inside_outside_function


def sample_points_on_predicted_shape(
    prim_params,
    sampler,
    sharpness,
    use_cuboid=False
):
    """
    Arguments:
    ----------
        prim_params: PrimitiveParameters object containing the predictions
                     of the network
    """
    # Declare some variables
    B = prim_params.batch_size
    M = prim_params.n_primitives
    S = sampler.n_samples

    probs = prim_params.probs
    translations = prim_params.translations_r
    rotations = prim_params.rotations_r
    alphas = prim_params.sizes_r
    epsilons = prim_params.shapes_r
    if prim_params.deformations is None:
        tapering_params = probs.new_zeros(B, M, 2)
    else:
        tapering_params = prim_params.tapering_params_r
    # Get the coordinates of the sampled points on the surfaces of the SQs,
    # with size BxMxSx3
    X_SQ, _ = sampler.sample_points_on_primitive(use_cuboid, alphas, epsilons)
    X_SQ = deform(X_SQ, alphas, tapering_params)
    # Make sure that everything has the right size
    assert X_SQ.shape == (B, M, S, 3)

    # Transform SQs to world coordinates
    X_SQ_world = transform_to_world_coordinates_system(
        X_SQ,
        translations,
        rotations
    )
    # Make sure that everything has the right size
    assert X_SQ_world.shape == (B, M, S, 3)
    # Transform the points on the SQs to the other SQs
    X_SQ_transformed = transform_to_primitives_centric_system(
        X_SQ_world.view(B, M*S, 3),
        translations.view(B, M, 3),
        rotations.view(B, M, 4)
    )
    assert X_SQ_transformed.shape == (B, M*S, M, 3)

    # Compute the inside outside function for every point on every primitive to
    # every other primitive
    F = inside_outside_function(
        X_SQ_transformed,
        alphas.detach(),  # numerical reasons for the detach ;)
        epsilons.view(B, M, 2).detach()
    )
    assert F.shape == (B, M*S, M)
    F = F.view(B, M, S, M)
    F = torch.sigmoid(sharpness*(1.0 - F))
    f = torch.max(F, dim=-1)[0]
    assert f.shape == (B, M, S)
    isolevel = F.new_tensor(0.49)
    mask = f <= isolevel

    assert B == 1
    X_SQ_mask = X_SQ_world[mask]
    assert len(X_SQ_mask.shape) == 2
    assert X_SQ_mask.shape[1] == 3
    return X_SQ_mask


def chamfer_loss(
    y_hat,
    y_target,
    sampler,
    options,
    use_l1=False
):
    """
    Implement the loss function using the implicit surface function of SQs

    Arguments:
    ----------
        y_hat: List of Tensors containing the predictions of the network
        y_target: Tensor with size BxNx6 with N points from the target object
                  and their corresponding normals
        options: A dictionary with various options

    Returns:
    -------
        the loss
    """
    sharpness = options.get("sharpness", 5.0)
    use_cuboid = options.get("use_cuboid", False)

    gt_points = y_target[:, :, :3]
    assert gt_points.shape[-1] == 3

    # Declare some variables
    N = gt_points.shape[1]  # number of points per sample

    X_SQ = sample_points_on_predicted_shape(
        y_hat, sampler, sharpness, use_cuboid
    )
    V = torch.abs(X_SQ.unsqueeze(0) - gt_points[0].unsqueeze(1))
    assert V.shape == (N, X_SQ.shape[0], 3)

    if use_l1:
        D = torch.sum(V, -1)
    else:
        D = torch.sum((V)**2, -1)

    D_pcl_to_prim = D.min(-1)[0].mean()
    D_prim_to_pcl = D.min(0)[0].mean()
    loss = D_pcl_to_prim + D_prim_to_pcl

    # Sum up the regularization terms
    return loss.mean()
