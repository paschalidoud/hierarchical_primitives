from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class Size(nn.Module):
    """Use the features to predict the size of all primitives.

    The shape of the Size tensor should be BxM*3, where B is the batch size and
    M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives, min_a=0.005, max_a=0.5):
        super(Size, self).__init__()
        self._n_primitives = n_primitives
        self.fc = nn.Linear(input_dims, n_primitives*3)
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, X, primitive_params):
        sizes = torch.sigmoid(self.fc(X)) * self.max_a + self.min_a

        return PrimitiveParameters.from_existing(
            primitive_params,
            sizes=sizes
        )


class DeepSize(nn.Module):
    """Use the features to predict the size of all primitives.

    The shape of the DeepSize tensor should be BxM*3, where B is the batch size
    and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives, min_a=0.005, max_a=0.5):
        super(DeepSize, self).__init__()
        self._n_primitives = n_primitives

        self.fc_0 = nn.Linear(input_dims, input_dims)
        self.nonlin_0 = nn.LeakyReLU(0.2, True)
        self.fc_1 = nn.Linear(input_dims, n_primitives*3)
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, X, primitive_params):
        sizes = torch.sigmoid(
            self.fc_1(self.nonlin_0(self.fc_0(X)))
        ) * self.max_a + self.min_a

        return PrimitiveParameters.from_existing(
            primitive_params,
            sizes=sizes
        )


class AttSize(nn.Module):
    def __init__(
        self, input_dims, n_layers, hidden_units, min_a=0.005, max_a=0.5
    ):
        super(AttSize, self).__init__()
        self.min_a = min_a
        self.max_a = max_a

        # Keep the layers based on the n_layers
        l = []
        in_features = input_dims
        for i in range(n_layers-1):
            l.append(nn.Linear(in_features, hidden_units))
            l.append(nn.ReLU())
            in_features = hidden_units
        l.append(nn.Linear(in_features, 3))
        self.fc = nn.Sequential(*l)

    def forward(self, X, primitive_params):
        sizes = torch.sigmoid(self.fc(X)) * self.max_a + self.min_a
        sizes = sizes.view(X.shape[0], -1)

        return PrimitiveParameters.from_existing(
            primitive_params,
            sizes=sizes
        )


def sizes(name, fe, n_primitives, config):
    layers = dict(
        default_size=partial(Size, fe.feature_shape, n_primitives),
        deep_size=partial(DeepSize, fe.feature_shape, n_primitives),
        att_size=partial(
            AttSize,
            fe.feature_shape,
            n_layers=config["data"].get("n_layers", 1),
            hidden_units=config["data"].get("hidden_units", 128)
        )
    )

    return layers[name]()
