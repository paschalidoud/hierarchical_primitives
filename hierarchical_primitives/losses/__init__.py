
from functools import partial

from ..utils.value_registry import ValueRegistry
from .implicit_surface_loss import implicit_surface_loss
from .loss_functions import euclidean_dual_loss
from .coverage import cluster_coverage_with_reconstruction
from .implicit_surface_loss_with_partition import implicit_surface_loss_with_partition
from .regularizers import get as _get_regularizer_function


def _get_loss_function(loss, sampler, options):
    if loss == "euclidean_dual_loss":
        return partial(
            euclidean_dual_loss,
            options=options,
            sampler=sampler
        )
    elif loss == "implicit_surface_loss":
        return partial(
            implicit_surface_loss,
            options=options
        )
    elif loss == "implicit_surface_loss_with_chamfer_loss":
        return partial(
            implicit_surface_loss_with_chamfer_loss,
            options=options,
            sampler=sampler
        )
    elif loss == "cluster_coverage":
        return partial(
            cluster_coverage_with_reconstruction,
            options=options
        )
    elif loss == "implicit_surface_loss_with_partition":
        return partial(
            implicit_surface_loss_with_partition,
            options=options
        )


def get_loss(loss, regularizers, sampler, options):
    loss_fn = _get_loss_function(loss, sampler, options)
    regularizers = [
        (_get_regularizer_function(regularizer, options), weight)
        for regularizer, weight in regularizers
    ]

    def inner(y_hat, y_target):
        ValueRegistry.get_instance("loss_intermediate_values").clear()
        loss = loss_fn(y_hat, y_target)
        for regularizer, weight in regularizers:
            loss = loss + weight*regularizer(y_hat)

        return loss

    return inner
