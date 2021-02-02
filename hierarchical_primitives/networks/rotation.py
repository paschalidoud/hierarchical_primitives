from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class Rotation(nn.Module):
    """Use the features to predict the rotation as a quaternion for all
    primitives.

    The shape of the Rotation tensor should be BxM*4, where B is the batch size
    and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(Rotation, self).__init__()
        self._n_primitives = n_primitives
        self.fc = nn.Linear(input_dims, n_primitives*4)

    def forward(self, X, primitive_params):
        quats = self.fc(X).view(X.shape[0], self._n_primitives, 4)
        # Apply an L2-normalization non-linearity to enforce the unit norm
        # constrain
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(X.shape[0], self._n_primitives*4)

        return PrimitiveParameters.from_existing(
            primitive_params,
            rotations=rotations
        )


class DeepRotation(nn.Module):
    """Use the features to predict the rotation as a quaternion for all
    primitives using a deeper architecture.

    The shape of the DeepRotation tensor should be BxM*4, where B is the batch
    size and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(DeepRotation, self).__init__()
        self._n_primitives = n_primitives

        self.fc_0 = nn.Linear(input_dims, input_dims)
        self.nonlin_0 = nn.LeakyReLU(0.2, True)
        self.fc_1 = nn.Linear(input_dims, n_primitives*4)

    def forward(self, X, primitive_params):
        quats = self.fc_1(
            self.nonlin_0(sle.fc_0(X))
        ).view(X.shape[0], self._n_primitives, 4)
        # Apply an L2-normalization non-linearity to enforce the unit norm
        # constrain
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(X.shape[0], self._n_primitives*4)

        return PrimitiveParameters.from_existing(
            primitive_params,
            rotations=rotations
        )


class AttRotation(nn.Module):
    """Use the features to predict the rotation for all primitives for an
    attention-based architecture.
    """
    def __init__(self, input_dims, n_layers, hidden_units):
        super(AttRotation, self).__init__()

        # Keep the layers based on the n_layers
        l = []
        in_features = input_dims
        for i in range(n_layers-1):
            l.append(nn.Linear(in_features, hidden_units))
            l.append(nn.ReLU())
            in_features = hidden_units
        l.append(nn.Linear(in_features, 4))
        self.fc = nn.Sequential(*l)

    def forward(self, X, primitive_params):
        quats = self.fc(X)
        # Apply an L2-normalization non-linearity to enforce the unit norm
        # constrain
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(X.shape[0], -1)

        return PrimitiveParameters.from_existing(
            primitive_params,
            rotations=rotations
        )


class NoRotation(nn.Module):
    def __init__(self, n_primitives):
        super(NoRotation, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        rotations = X.new_zeros(X.shape[0], self._n_primitives, 4)
        rotations[:, :, 0] = 1.0

        return PrimitiveParameters.from_existing(
            primitive_params,
            rotations=rotations.view(X.shape[0], self._n_primitives*4)
        )


def rotations(name, fe, n_primitives, config):
    layers = dict(
        default_rotation=partial(Rotation, fe.feature_shape, n_primitives),
        no_rotation=partial(NoRotation, n_primitives),
        deep_rotation=partial(DeepRotation, fe.feature_shape, n_primitives),
        att_rotation=partial(
            AttRotation,
            fe.feature_shape,
            n_layers=config["data"].get("n_layers", 1),
            hidden_units=config["data"].get("hidden_units", 128)
        )
    )
    return layers[name]()
