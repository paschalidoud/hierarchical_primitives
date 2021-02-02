from functools import partial

import torch

from ..utils.stats_logger import StatsLogger
from ..utils.value_registry import ValueRegistry
from ..primitives import sq_volumes


def volumes(parameters):
    """Ensure that the primitives will be small"""
    volumes = sq_volumes(parameters)
    return volumes.mean()


def sparsity(parameters, minimum_number_of_primitives,
              maximum_number_of_primitives, w1, w2, a1, a2):
    """Ensure that we have at least that many primitives in expectation"""
    expected_primitives = parameters[0].sum(-1)

    lower_bound = minimum_number_of_primitives - expected_primitives
    upper_bound = expected_primitives - maximum_number_of_primitives
    zero = expected_primitives.new_tensor(0)

    t1 = torch.max(lower_bound, zero) * lower_bound**a1
    t2 = torch.max(upper_bound, zero) * upper_bound**a2

    return (w1*t1 + w2*t2).mean()


def entropy_bernoulli(parameters):
    """Minimize the entropy of each bernoulli variable pushing them to either 1
    or 0"""
    probs = parameters[0]
    sm = probs.new_tensor(1e-3)

    t1 = torch.log(torch.max(probs, sm))
    t2 = torch.log(torch.max(1 - probs, sm))

    return torch.mean((-probs * t1 - (1-probs) * t2).sum(-1))


def parsimony(parameters):
    """Penalize the use of more primitives"""
    expected_primitives = parameters[0].sum(-1)

    return expected_primitives.mean()


def siblings_proximity(parameters, root_depth=1, maximum_distance=0.1):
    """Make sure that two primitives that have the same parent will also be
       close in space.
    """
    _, P = parameters.space_partition
    zero = P[0].translations.new_tensor(0.0)
    max_depth = len(P)
    D = 0
    # Iterate over the depths
    for d in range(root_depth, max_depth):
        t = P[d].translations_r
        t1 = t[:, 0::2]
        t2 = t[:, 1::2]
        D = D + torch.max(
            torch.sqrt(torch.sum((t1-t2)**2, dim=-1)) - maximum_distance,
            zero
        ).sum()
    N = P[0].translations.new_tensor(
            (2**torch.arange(root_depth-1, max_depth-1)).sum()
        ).float()
    return D / N


def overlapping(parameters, tau=2):
    """Make sure that at most tau primitives witll overlap
    """
    intermediate_values = ValueRegistry.get_instance(
        "loss_intermediate_values"
    )
    F = intermediate_values["F"]
    probs = parameters.probs
    # Make sure that everything has the right size
    assert probs.shape[0] == F.shape[0]
    assert probs.shape[1] == F.shape[2]
    zero = probs.new_tensor(0.0)

    # Only consider primitives that exist
    t = (probs.unsqueeze(1) * F).sum(dim=-1)
    return torch.max(zero, t - tau).mean()


def overlapping_on_depths(parameters, tau=1):
    intermediate_values = ValueRegistry.get_instance(
        "loss_intermediate_values"
    )
    F_intermediate = intermediate_values["F_intermediate"]
    _, P = parameters.space_partition
    zero = parameters.probs.new_tensor(0.0)

    reg_terms = []
    for pcurrent, F in zip(P[1:], F_intermediate[1:]):
        probs = pcurrent.probs
        # Make sure that everything has the right size
        assert probs.shape[0] == F.shape[0]
        assert probs.shape[1] == F.shape[2]

        # Only consider primitives that exist
        mask = (F >= 0.5).float()
        t = (probs.unsqueeze(1) * mask * F).sum(dim=-1)
        reg_terms.append(torch.max(zero, t - tau).mean()/pcurrent.n_primitives)

    return sum(reg_terms)


def get(regularizer, options):
    regs = {
        "parsimony": parsimony,
        "entropy_bernoulli": entropy_bernoulli,
        "overlapping": lambda y_hat: overlapping(
            y_hat,
            options.get("tau", 2)
        ),
        "overlapping_on_depths": overlapping_on_depths,
        "sparsity": lambda y_hat: sparsity(
            y_hat,
            options.get("minimum_number_of_primitives", 5),
            options.get("maximum_number_of_primitives", 5000),
            options.get("w1", 0.005),
            options.get("w2", 0.005),
            options.get("a1", 4.0),
            options.get("a2", 2.0)
        ),
        "proximity": proximity, 
        "siblings_proximity": siblings_proximity,
        "volumes": volumes
    }

    def inner(y_hat):
        reg_value = regs[regularizer](y_hat)
        reg_key = "regularizers." + regularizer
        StatsLogger.instance()[reg_key] = reg_value.item()

        return reg_value

    return inner
