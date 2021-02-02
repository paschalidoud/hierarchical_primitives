from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class QualityOfSuperquadric(nn.Module):
    def __init__(self, input_dims, n_layers, hidden_units):
        super(QualityOfSuperquadric, self).__init__()
        # Keep the layers based on the n_layers
        l = []
        in_features = input_dims
        for i in range(n_layers-1):
            l.append(nn.Linear(in_features, hidden_units))
            l.append(nn.ReLU())
            in_features = hidden_units
        l.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*l)


    def forward(self, X, primitive_params):
        qos = torch.sigmoid(self.fc(X)).squeeze(-1)
        return PrimitiveParameters.from_existing(
            primitive_params,
            qos=qos
        )

def qos(name, fe, n_primitives, config):
    layers = dict(
        att_qos=partial(
            QualityOfSuperquadric,
            fe.feature_shape,
            n_layers=config["data"].get("n_layers", 1),
            hidden_units=config["data"].get("hidden_units", 128)
        )
    )
    return layers[name]()
