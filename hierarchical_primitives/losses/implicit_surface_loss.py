import torch

from .common import loss_weights_on_depths
from .loss_functions import euclidean_dual_loss, _euclidean_dual_loss_impl
from ..primitives import get_implicit_surface, _accuracy_and_recall

from ..utils.metrics import compute_iou
from ..utils.stats_logger import StatsLogger
from ..utils.value_registry import ValueRegistry


def _implicit_surface_loss_impl(X, X_weights, labels, probs, F):
    # Get the sizes: batch size (B), number of points (N) and number of
    # primitives
    B, N, M = F.shape
    assert X.shape == (B, N, 3)
    assert X_weights.shape == (B, N, 1)
    assert labels.shape == (B, N, 1)

    # Sort F in descending order
    f, idxs = torch.sort(F, dim=-1, descending=True)

    # Start by computing the cumulative product
    # Sort based on the indices
    probs = torch.cat([
        probs[i].take(idxs[i]).unsqueeze(0) for i in range(len(idxs))
    ])
    neg_cumprod = torch.cumprod(1-probs, dim=-1)
    neg_cumprod = torch.cat(
        [neg_cumprod.new_ones((B, N, 1)), neg_cumprod[:, :, :-1].clone()],
        dim=-1
    )

    # minprob[i, j, k] is the probability that for sample i and point j the
    # k-th primitive has the minimum loss
    minprob = probs.mul(neg_cumprod)

    intermediate_values = ValueRegistry.get_instance(
        "loss_intermediate_values"
    )
    intermediate_values["F_sorted"] = f
    intermediate_values["minprob"] = minprob

    # Compute the classification loss using binary cross entropy loss
    sm = probs.new_tensor(1e-6)
    t1 = torch.log(torch.max(f, sm))
    t2 = torch.log(torch.max(1.0 - f, sm))
    cross_entropy_loss = - labels * t1 - (1.0 - labels) * t2
    cross_entropy_loss = X_weights * cross_entropy_loss
    loss = torch.einsum("ijk,ijk->i", [cross_entropy_loss, minprob])
    loss = loss / N

    return loss


def implicit_surface_loss(prim_params, y_target, options):
    """
    Implement the loss function using the implicit surface function of SQs

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
    gt_points, gt_labels, gt_weights = y_target

    # Declare some variables
    B = gt_points.shape[0]  # batch size
    N = gt_points.shape[1]  # number of points per sample
    M = prim_params.n_primitives  # number of primitives

    probs = prim_params.probs
    translations = prim_params.translations_r
    rotations = prim_params.rotations_r
    alphas = prim_params.sizes_r
    epsilons = prim_params.shapes_r
    sharpness = prim_params.sharpness_r

    # Compute the implicit surface function for each primitive
    F, X_transformed = get_implicit_surface(
        gt_points, translations, rotations, alphas, epsilons, sharpness
    )
    intermediate_values = ValueRegistry.get_instance(
        "loss_intermediate_values"
    )
    intermediate_values["F"] = F
    intermediate_values["X"] = X_transformed
    assert F.shape == (B, N, M)

    loss = _implicit_surface_loss_impl(
        gt_points, gt_weights, gt_labels, probs, F
    )

    # Compute some metrics to report during training
    iou = compute_iou(
        gt_labels.squeeze(-1),
        torch.max(F, dim=-1)[0]
    )

    accuracy, positive_accuracy = _accuracy_and_recall(
        intermediate_values["F_sorted"],
        intermediate_values["minprob"],
        gt_labels,
        gt_weights
    )
    stats = StatsLogger.instance()
    stats["losses.reconstruction"] = loss.mean().item()
    stats["metrics.accuracy"] = accuracy.item()
    stats["metrics.positive_accuracy"] = positive_accuracy.item()
    stats["metrics.iou"] = iou
    stats["metrics.exp_n_prims"] = probs.sum(-1).mean().item()

    return loss.mean()
