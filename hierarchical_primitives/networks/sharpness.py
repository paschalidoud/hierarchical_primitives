from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class ConstantSharpness(nn.Module):
    def __init__(self, sharpness_value):
        super(ConstantSharpness, self).__init__()
        self.sv = sharpness_value

    def forward(self, X, primitive_params):
        sharpness = X.new_ones(
            (X.shape[0], primitive_params.n_primitives, 2)
        ) * X.new_tensor(self.sv)
        
        return PrimitiveParameters.from_existing(
            primitive_params,
            sharpness=sharpness
        )


class MixedConstantSharpness(nn.Module):
    def __init__(self, sharpness_value_pos, sharpness_value_neg):
        super(MixedConstantSharpness, self).__init__()
        self.sv_pos = sharpness_value_pos
        self.sv_neg = sharpness_value_neg

    def forward(self, X, primitive_params):
        sharpness = X.new_ones(
            (X.shape[0], primitive_params.n_primitives, 2)
        )
        sharpness[:, :, 0] = self.sv_pos
        sharpness[:, :, 1] = self.sv_neg
        
        return PrimitiveParameters.from_existing(
            primitive_params,
            sharpness=sharpness
        )


class Sharpness(nn.Module):
    def __init__(self, input_dims, n_primitives, max_sv=10.0):
        super(Sharpness, self).__init__()
        self.max_sv = max_sv
        self.fc = nn.Linear(input_dims, n_primitives)

    def forward(self, X, primitive_params):
        sv = torch.sigmoid(self.fc(X)) * self.max_sv
        return PrimitiveParameters.from_existing(
            primitive_params,
            sharpness=sv.unsqueeze(-1).expand(X.shape[0], -1, 2)
        )


class MixedSharpness(nn.Module):
    def __init__(
        self, input_dims, n_primitives, max_sv_pos=10.0, max_sv_neg=10.0
    ):
        super(MixedSharpness, self).__init__()
        self.max_sv_pos = max_sv_pos
        self.max_sv_neg = max_sv_neg
        self.fc = nn.Linear(input_dims, 2*n_primitives)

    def forward(self, X, primitive_params):
        s = torch.sigmoid(self.fc(X)).view(X.shape[0], -1, 2)
        s = s * s.new_tensor([[[self.max_sv_pos, self.max_sv_neg]]])

        return PrimitiveParameters.from_existing(
            primitive_params,
            sharpness=s
        )


def sharpness(name, fe, n_primitives, config):
    layers = dict(
        constant_sharpness=partial(
            ConstantSharpness,
            sharpness_value=config["loss"].get("sharpness", 10.0)
        ),
        mixed_constant_sharpness=partial(
            MixedConstantSharpness,
            sharpness_value_pos=config["loss"].get("sharpness", 10.0),
            sharpness_value_neg=config["loss"].get("sharpness_neg", 10.0)
        ),
        sharpness=partial(
            Sharpness,
            fe.feature_shape,
            n_primitives,
            max_sv=config["loss"].get("sharpness", 10.0)
        ),
        mixed_sharpness=partial(
            MixedSharpness,
            fe.feature_shape,
            n_primitives,
            max_sv_pos=config["loss"].get("sharpness", 10.0),
            max_sv_neg=config["loss"].get("sharpness_neg", 10.0)
        ),
    )

    return layers[name]()

