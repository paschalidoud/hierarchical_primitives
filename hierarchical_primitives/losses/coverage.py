
import torch

from .regularizers import overlapping_on_depths
from ..networks.primitive_parameters import PrimitiveParameters
from ..primitives import get_implicit_surface, _compute_accuracy_and_recall
from ..utils.stats_logger import StatsLogger
from ..utils.value_registry import ValueRegistry
from ..utils.metrics import compute_iou


def cluster_coverage_with_reconstruction(prim_params, y_target, options):
    def _coverage_inner(p, pparent, X, labels):
        M = p.n_primitives  # number of primitives
        B, N, _ = X.shape
        translations = p.translations_r
        splits = 2 if M > 1 else 1
        assert labels.shape == (B, N, M//splits)

        # First assign points from the labels to each of the siblings
        dists = ((X.unsqueeze(2) - translations.unsqueeze(1))**2).sum(-1)
        assert dists.shape == (B, N, M)
        if M > 1:
            assign_left = (dists[:, :, ::2] < dists[:, :, 1::2]).float()
            assign_right = 1-assign_left
            assignments = torch.stack([
                assign_left * labels,
                assign_right * labels
            ], dim=-1).view(B, N, M)
        else:
            assignments = labels
        assert assignments.shape == (B, N, M)
        assert assignments.sum(-1).max().item() == 1

        # Now compute the sum of squared distances as the loss
        loss = (dists * assignments).sum(-1).mean()

        return loss, assignments

    def _fit_shape_inner(pr, X, X_labels, X_weights):
        M = pr.n_primitives  # number of primitives
        B, N, _ = X.shape

        assert X_labels.shape == (B, N, 1)
        assert X_weights.shape == (B, N, 1)

        translations = pr.translations_r
        rotations = pr.rotations_r
        alphas = pr.sizes_r
        epsilons = pr.shapes_r
        sharpness = pr.sharpness_r

        # Compute the implicit surface function for each primitive
        F, _ = get_implicit_surface(
            X, translations, rotations, alphas, epsilons, sharpness
        )
        assert F.shape == (B, N, M)

        f = torch.max(F, dim=-1, keepdim=True)[0]
        sm = F.new_tensor(1e-6)
        t1 = torch.log(torch.max(f, sm))
        t2 = torch.log(torch.max(1.0 - f, sm))
        cross_entropy_loss = - X_labels * t1 - (1.0 - X_labels) * t2
        loss = X_weights * cross_entropy_loss

        return loss.mean(), F

    def _fit_parent_inner(p, X, labels, X_weights, F):
        M = p.n_primitives  # number of primitives
        B, N, _ = X.shape

        assert labels.shape == (B, N, M)

        translations = p.translations_r
        rotations = p.rotations_r
        alphas = p.sizes_r
        epsilons = p.shapes_r
        sharpness = p.sharpness_r

        sm = F.new_tensor(1e-6)
        t1 = labels * torch.log(torch.max(F, sm))
        t2 = (1-labels) * torch.log(torch.max(1-F, sm))
        ce = - t1 - t2
        # 5 is a very important number that is necessary for the code to work!!!
        # Do not change!! (This is there to avoid having empty primitives :-))
        loss_mask = (labels.sum(1, keepdim=True) > 5).float()
        loss = (loss_mask*ce*X_weights).mean()

        # Compute the quality of the current SQ
        target_iou = compute_iou(
            F.transpose(2, 1).reshape(-1, N),
            labels.transpose(2, 1).reshape(-1, N),
            average=False
        ).view(B, M).detach()
        mse_qos_loss = ((p.qos - target_iou)**2).mean()

        return loss, mse_qos_loss, F

    # Extract the arguments to local variables
    gt_points, gt_labels, gt_weights = y_target
    _, P = prim_params.space_partition
    sharpness = prim_params.sharpness_r

    # Compute the coverage loss given the partition
    labels = [gt_labels]
    coverage_loss = 0
    for i in range(len(P)):
        pcurrent = P[i]
        if i == 0:
            precision_m = gt_points.new_zeros(3, 3).fill_diagonal_(1).reshape(
                1, 3, 3).repeat((gt_points.shape[0], 1, 1, 1)
            )
            pparent = PrimitiveParameters.from_existing(
                PrimitiveParameters.empty(),
                precision_matrix=precision_m
            )
        else:
            pparent = P[i-1]
        loss, next_labels = _coverage_inner(
            pcurrent, pparent, gt_points, labels[-1]
        )
        labels.append(next_labels)
        coverage_loss = coverage_loss + loss

    F_intermediate = []
    fit_loss = 0
    pr_loss = 0
    for pr, ps in zip(prim_params.fit, P):
        floss, F = _fit_shape_inner(pr, gt_points, gt_labels, gt_weights)
        fit_loss = fit_loss + 1e-1 * floss
        F_intermediate.append(F)

        # Compute the proximity loss between the centroids and the centers of the
        # primitives
        s_tr = ps.translations_r.detach()
        r_tr = pr.translations_r
        pr_loss = pr_loss + ((s_tr - r_tr)**2).sum(-1).mean()

    # Compute the disjoint loss between the siblings
    intermediates = ValueRegistry.get_instance("loss_intermediate_values")
    intermediates["F_intermediate"] = F_intermediate
    intermediates["labels"] = labels
    intermediates["gt_points"] = gt_points

    # Compute the quality of the reconstruction
    qos_loss = 0
    for i, pr in enumerate(prim_params.fit):
        floss, qloss, F = _fit_parent_inner(
            pr, gt_points, labels[i+1], gt_weights, F_intermediate[i]
        )
        fit_loss = fit_loss + 1e-2 * floss
        qos_loss = qos_loss + 1e-3 * qloss

    # Compute some metrics to report during training
    F_leaves = F_intermediate[-1]
    iou = compute_iou(
        gt_labels.squeeze(-1),
        torch.max(F_leaves, dim=-1)[0]
    )
    accuracy, positive_accuracy = _compute_accuracy_and_recall(
        F_leaves,
        F_leaves.new_ones(F_leaves.shape[0], F_leaves.shape[-1]),
        gt_labels,
        gt_weights
    )

    stats = StatsLogger.instance()
    stats["losses.coverage"] = coverage_loss.item()
    stats["losses.fit"] = fit_loss.item()
    stats["losses.prox"] = pr_loss.item()
    stats["losses.qos"] = qos_loss.item()
    stats["metrics.iou"] = iou
    stats["metrics.accuracy"] = accuracy.item()
    stats["metrics.positive_accuracy"] = positive_accuracy.item()

    return coverage_loss + pr_loss + fit_loss + qos_loss
