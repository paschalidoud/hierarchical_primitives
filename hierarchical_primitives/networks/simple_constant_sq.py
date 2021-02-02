from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class SimpleConstantSQ(nn.Module):
    def __init__(self, n_primitives):
        super(SimpleConstantSQ, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        B = X.shape[0]
        M = self._n_primitives
        rotations = X.new_zeros(B, M*4)
        rotations[:, ::4] = 1.
        return PrimitiveParameters.from_existing(
            primitive_params,
            sizes=X.new_ones(B, M*3)*0.1,
            shapes=X.new_ones(B, M*2),
            rotations=rotations
        )

def simple_constant_sq(name, fe, n_primitives, config):
    layers = dict(
        default_sq=partial(SimpleConstantSQ, n_primitives),
    )
    return layers[name]()
