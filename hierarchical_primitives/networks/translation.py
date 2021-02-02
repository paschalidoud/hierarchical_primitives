from functools import partial

import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters


class Translation(nn.Module):
    """Use the features to predict the translation vectors for all primitives.

    The shape of the Translation tensor should be BxM*3, where B is the batch
    size and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(Translation, self).__init__()
        self.fc = nn.Linear(input_dims, n_primitives*3)

    def forward(self, X, primitive_params):
        # Everything lies in the unit cube, so the maximum translation vector
        # is 0.51
        translations = torch.tanh(self.fc(X)) * 0.51

        return PrimitiveParameters.from_existing(
            primitive_params,
            translations=translations
        )


class DeepTranslation(nn.Module):
    """Use the features to predict the translation vectors for all primitives
    using a deeper architecture.

    The shape of the Translation tensor should be BxM*3, where B is the batch
    size and M is the number of primitives.
    """
    def __init__(self, input_dims, n_primitives):
        super(DeepTranslation, self).__init__()
        self.fc_0 = nn.Linear(input_dims, input_dims)
        self.nonlin_0 = nn.LeakyReLU(0.2, True)
        self.fc_1 = nn.Linear(input_dims, n_primitives*3)

    def forward(self, X, primitive_params):
        # Everything lies in the unit cube, so the maximum translation vector
        # is 0.51
        translations = torch.tanh(
            self.fc_1(self.nonlin_0(self.fc_0(X)))
        ) * 0.51

        return PrimitiveParameters.from_existing(
            primitive_params,
            translations=translations
        )


class AttTranslation(nn.Module):
    def __init__(self, input_dims, n_layers, hidden_units):
        super(AttTranslation, self).__init__()

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
        translations = torch.tanh(self.fc(X)) * 0.51

        # Reshape to BxM*3
        translations = translations.view(X.shape[0], -1)
        return PrimitiveParameters.from_existing(
            primitive_params,
            translations=translations
        )


class RelativeTranslation(nn.Module):
    def __init__(self, n_primitives):
        super(RelativeTranslation, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        # Get the translations, the probs and the Pi_n tensors from the
        # primitive_params
        _translations = primitive_params.translations
        _probs = primitive_params.probs
        _Pi_n = primitive_params.Pi_n

        # Denote some variables for convenience
        B = X.shape[0]
        M = self._n_primitives

        # Compute the global translations from the local ones
        mask = X.new_tensor(1.) - torch.eye(M).to(X.device)
        g_translations = _translations.view(B, M, -1)
        for P in _Pi_n:
            g_translations = (
                g_translations +
                torch.einsum(
                    "ikc,ijk,ijl->ijl",
                    [
                        _probs.unsqueeze(-1),
                        P*mask,
                        _translations.view(B, M, -1)
                    ]
                )
            )

        return PrimitiveParameters.from_existing(
            primitive_params,
            translations=g_translations,
            local_translations=_translations
        )


class NoTranslation(nn.Module):
    """By default the translation tensor is set to 0.0."""
    def __init__(self, n_primitives):
        super(NoTranslation, self).__init__()
        self._n_primitives = n_primitives

    def forward(self, X, primitive_params):
        translations = X.new_zeros(X.shape[0], self._n_primitives*3)

        return PrimitiveParameters.from_existing(
            primitive_params,
            translations=translations
        )


def translations(name, fe, n_primitives, config):
    layers = dict(
        default_translation=partial(
            Translation,
            fe.feature_shape,
            n_primitives
        ),
        deep_translation=partial(
            DeepTranslation,
            fe.feature_shape,
            n_primitives
        ),
        att_translation=partial(
            AttTranslation,
            fe.feature_shape,
            n_layers=config["data"].get("n_layers", 1),
            hidden_units=config["data"].get("hidden_units", 128)
        ),
        no_translation=partial(NoTranslation, n_primitives),
        relative_translation=partial(RelativeTranslation, n_primitives)
    )
    return layers[name]()
