from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class SQShape(nn.Module):
    """Use the features to predict the shape for all primitives.

    The shape of the SQShape tensor should be BxM*2, where B is the batch size
    and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives, min_e=0.4, max_e=1.1):
        super(SQShape, self).__init__()
        self.fc = nn.Linear(input_dims, n_primitives*2)
        self.max_e = max_e
        self.min_e = min_e

    def forward(self, X, primitive_params):
        shapes = torch.sigmoid(self.fc(X)) * self.max_e + self.min_e

        return PrimitiveParameters.from_existing(
            primitive_params,
            shapes=shapes
        )


class DeepSQShape(nn.Module):
    """Use the features to predict the shape for all primitives using a deeper
    architecture.

    The shape of the DeepSQShape tensor should be BxM*2, where B is the batch
    size and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives, min_e=0.4, max_e=1.1):
        super(DeepSQShape, self).__init__()
        self.fc_0 = nn.Linear(input_dims, input_dims)
        self.nonlin_0 = nn.LeakyReLU(0.2, True)
        self.fc_1 = nn.Linear(input_dims, n_primitives*2)
        self.max_e = max_e
        self.min_e = min_e

    def forward(self, X, primitive_params):
        shapes = torch.sigmoid(
            self.fc_1(self.nonlin_0(self.fc_0(X)))
        ) * self.max_e + self.min_e

        return PrimitiveParameters.from_existing(
            primitive_params,
            shapes=shapes
        )


class CubeShape(nn.Module):
    """By default all primitives are cubes, thus their shape is 0.25"""
    def __init__(self, n_primitives):
        super(CubeShape, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        # Shapes should have shape BxM*2
        shapes = X.new_ones((X.shape[0], self._n_primitives*2)) * 0.25

        return PrimitiveParameters.from_existing(
            primitive_params,
            shapes=shapes
        )


class AttSQShape(nn.Module):
    """Use the features to predict the shape for all primitives.

    The shape of the AttentionSQShape tensor should be BxM*2, where B is the
    batch size and M is the number of primitives.
    """
    def __init__(
        self, input_dims, n_layers, hidden_units, min_e=0.4, max_e=1.1
    ):
        super(AttSQShape, self).__init__()
        self.max_e = max_e
        self.min_e = min_e

        # Keep the layers based on the n_layers
        l = []
        in_features = input_dims
        for i in range(n_layers-1):
            l.append(nn.Linear(in_features, hidden_units))
            l.append(nn.ReLU())
            in_features = hidden_units
        l.append(nn.Linear(in_features, 2))
        self.fc = nn.Sequential(*l)

    def forward(self, X, primitive_params):
        shapes = torch.sigmoid(self.fc(X)) * self.max_e + self.min_e
        shapes = shapes.view(X.shape[0], -1)

        return PrimitiveParameters.from_existing(
            primitive_params,
            shapes=shapes
        )


def shapes(name, fe, n_primitives, config):
    layers = dict(
        sq=partial(SQShape, fe.feature_shape, n_primitives),
        cuboid=partial(CubeShape, n_primitives),
        deep_sq=partial(DeepSQShape, fe.feature_shape, n_primitives),
        att_sq=partial(
            AttSQShape,
            fe.feature_shape,
            n_layers=config["data"].get("n_layers", 1),
            hidden_units=config["data"].get("hidden_units", 128)
        )
    )
    return layers[name]()
