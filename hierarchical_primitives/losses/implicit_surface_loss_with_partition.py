import torch

from ..primitives import get_implicit_surface, _compute_accuracy_and_recall
from ..utils.stats_logger import StatsLogger
from ..utils.metrics import compute_iou


def implicit_surface_loss_with_partition(prim_params, y_target, options):
    """
    Arguments:
    ----------
        prim_params: PrimitiveParameters object containing the predictions
                     of the network
        y_target: A tensor of shape BxNx4 containing the points and occupancy
                  labels concatenated in the last dimension [x_i; o_i]
        options: A dictionary with various options

    Returns:
    -------
        the loss
    """
    def _structure_loss_inner(p, X, X_labels):
        M = p.n_primitives  # number of primitives
        B, N, _ = X.shape
        translations = p.translations_r

        # Compute the euclidean distance between the geometric centroids and
        # the target points
        dists = ((X.unsqueeze(2) - translations.unsqueeze(1))**2).sum(-1)
        assert dists.shape == (B, N, M)
        # For every point in the target points find the index of its closest
        # geometric centroid and assign it to this centroid.
        min_dists, idxs = torch.min(dists, dim=-1)
        X_labels_new = torch.eye(M, device=X.device)[idxs] * X_labels
        assert X_labels_new.shape == (B, N, M)
        assert X_labels_new.sum(-1).max().item() == 1

        # Now compute the sum of squared distances as the loss
        loss = (min_dists * X_labels.squeeze(-1)).mean()
        return loss, X_labels_new

    def _fit_part_inner(F, X, labels, X_weights):
        B, N, M = F.shape
        assert X_weights.shape == (B, N, 1)
        assert labels.shape == (B, N, M)

        # Now compute the loss
        sm = F.new_tensor(1e-6)
        t1 = torch.log(torch.max(F, sm))
        t2 = torch.log(torch.max(1-F, sm))
        loss = - labels * t1 - (1.0 - labels) * t2
        # 5 is a very important number that is necessary for the code to work!!!
        # Do not change!! (This is there to avoid having empty primitives :-))
        loss_mask = (labels.sum(1, keepdim=True) > 5).float()
        loss = (loss_mask*loss*X_weights).mean()

        return loss.mean()

    def _fit_shape_inner(F, X, X_labels, X_weights):
        B, N, M = F.shape
        assert X_labels.shape == (B, N, 1)
        assert X_weights.shape == (B, N, 1)

        # Simply compute the cross entropy loss
        f = torch.max(F, dim=-1, keepdim=True)[0]
        sm = F.new_tensor(1e-6)
        t1 = torch.log(torch.max(f, sm))
        t2 = torch.log(torch.max(1.0 - f, sm))
        cross_entropy_loss = - X_labels * t1 - (1.0 - X_labels) * t2
        loss = X_weights * cross_entropy_loss

        return loss.mean(), f

    # Extract the arguments to local variables
    X, X_labels, X_weights = y_target
    _, P = prim_params.space_partition

    # Compute the structure loss and the assign the labels on the points given
    # the partition
    structure_loss, new_labels = _structure_loss_inner(P[-1], X, X_labels)

    # Declare some variables
    M = prim_params.n_primitives
    B, N, _ = X.shape
    translations = prim_params.translations_r
    rotations = prim_params.rotations_r
    alphas = prim_params.sizes_r
    epsilons = prim_params.shapes_r
    sharpness = prim_params.sharpness_r
    # Compute the implicit surface function for each primitive
    F, _ = get_implicit_surface(
        X, translations, rotations, alphas, epsilons, sharpness
    )
    assert F.shape == (B, N, M)

    # Compute the geometry loss
    # Fit every primitive to the part of the object it represents
    fit_loss_parts = _fit_part_inner(F, X, new_labels, X_weights)

    fit_loss_shape, F_max = _fit_shape_inner(F, X, X_labels, X_weights)

    # Compute the proximity loss between the centroids and the centers of the
    # primitives
    s_tr = P[-1].translations_r.detach()
    r_tr = prim_params.translations_r
    prox_loss = ((s_tr - r_tr)**2).sum(-1).mean()

    # Compute some metrics and report during training
    iou = compute_iou(
        X_labels.squeeze(-1), F_max.squeeze(-1), X_weights.squeeze(-1)
    )
    accuracy, positive_accuracy = _compute_accuracy_and_recall(
        F,
        F.new_ones(F.shape[0], F.shape[-1]),
        X_labels,
        X_weights
    )

    stats = StatsLogger.instance()
    stats["losses.structure"] = structure_loss.item()
    stats["losses.fit_parts"] = fit_loss_parts.item()
    stats["losses.fit_shape"] = fit_loss_shape.item()
    stats["losses.prox"] = prox_loss.item()
    stats["metrics.iou"] = iou
    stats["metrics.accuracy"] = accuracy.item()
    stats["metrics.positive_accuracy"] = positive_accuracy.item()

    w1 = options["loss_weights"].get("structure_loss_weight", 0.0)
    w2 = options["loss_weights"].get("fit_shape_loss_weight", 0.0)
    w3 = options["loss_weights"].get("fit_parts_loss_weight", 0.0)
    w4 = options["loss_weights"].get("proximity_loss_weight", 0.0)

    loss = w1 * structure_loss + w2 * fit_loss_shape + w3 * fit_loss_parts
    loss = loss + w4 * prox_loss

    return loss
