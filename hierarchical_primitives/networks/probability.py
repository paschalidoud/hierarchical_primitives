from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class Probability(nn.Module):
    """Use the features to predict the existence probabilities for all the
    primitives.

    The shape of the Probability tensor shoud be BxM, where B is the batch size
    and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(Probability, self).__init__()
        self.fc = nn.Linear(input_dims, n_primitives)

    def forward(self, X, primitive_params):
        probs = torch.sigmoid(self.fc(X))

        return PrimitiveParameters.from_existing(
            primitive_params,
            probs=probs
        )


class DeepProbability(nn.Module):
    """Use the features to predict the existence probabilities for all the
    primitives with a deeper architecture.

    The shape of the Probability tensor shoud be BxM, where B is the batch size
    and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(DeepProbability, self).__init__()
        self.fc_0 = nn.Linear(input_dims, input_dims)
        self.nonlin_0 = nn.LeakyReLU(0.2, True)
        self.fc_1 = nn.Linear(input_dims, n_primitives)

    def forward(self, X, primitive_params):
        probs = torch.sigmoid(
            self.fc_1(self.nonlin_0(self.fc_0(X)))
        )

        return PrimitiveParameters.from_existing(
            primitive_params,
            probs=probs
        )


class AttProbability(nn.Module):
    """Use the features to predict the existence probabilities for all the
    primitives."""
    def __init__(self, input_dims):
        super(AttProbability, self).__init__()
        self.fc = nn.Linear(input_dims, 1)

    def forward(self, X, primitive_params):
        # Reshape it to BxM to be compatible with the rest
        probs = torch.sigmoid(self.fc(X)).squeeze(-1)

        return PrimitiveParameters.from_existing(
            primitive_params,
            probs=probs
        )


class AllOnes(nn.Module):
    """By default all primitives exist thus existence probabilities are 1."""
    def forward(self, X, primitive_params):
        probs = X.new_ones((X.shape[0], primitive_params.n_primitives))

        return PrimitiveParameters.from_existing(
            primitive_params,
            probs=probs
        )


class ProbabilityFromTransition(nn.Module):
    def __init__(self, n_primitives):
        super(ProbabilityFromTransition, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        _transitions = primitive_params.transitions
        probs = transitions[:, :self._n_primitives, :self._n_primitives]
        
        return PrimitiveParameters.from_existing(
            primitive_params,
            probs=probs
        )


class TerminationProbability(nn.Module):
    """Use the features to predict the termination probabilities for all the
    primitives."""
    def __init__(self, input_dims):
        super(TerminationProbability, self).__init__()
        self.fc = nn.Linear(input_dims, 1)

    def forward(self, X, primitive_params):
        ones = X.new_ones(X.shape[0], 1)
        probs = torch.sigmoid(self.fc(X)).squeeze(-1)
        probs = torch.cat([probs[:, :-1], ones], dim=1)

        return PrimitiveParameters.from_existing(
            primitive_params,
            termination_probs=probs
        )


def probs(name, fe, n_primitives, config):
    layers = dict(
        prob=partial(Probability, fe.feature_shape, n_primitives),
        all_ones=AllOnes,
        deep_prob=partial(DeepProbability, fe.feature_shape, n_primitives),
        att_prob=partial(AttProbability, fe.feature_shape),
        termination_prob=partial(TerminationProbability, fe.feature_shape),
        prob_from_transition=partial(ProbabilityFromTransition, n_primitives)
    )

    return layers[name]()
