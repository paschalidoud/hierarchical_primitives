import torch
import torch.nn as nn

from .primitive_parameters import PrimitiveParameters
from .probability import probs
from .rotation import rotations
from .shape import shapes
from .size  import sizes
from .translation import translations
from .qos import qos
from .sharpness import sharpness
from .simple_constant_sq import simple_constant_sq

from ..primitives import compose_quaternions


class _BasePrimitiveNetwork(nn.Module):
    """A simple way to reuse functions between primitive networks."""
    def _get_inputs(self, X):
        if isinstance(X, tuple):
            X, p = X
        else:
            p = PrimitiveParameters.empty()

        return X, p


class PrimitiveNetwork(_BasePrimitiveNetwork):
    """A PrimitiveNetwork creates a PrimitiveParameters object and passes it to
    a set of layers to fill it."""
    def __init__(self, layers):
        super(PrimitiveNetwork, self).__init__()
        for i, l in enumerate(layers):
            self.add_module("layer{}".format(i), l)
        self._layers = layers

    def forward(self, X):
        X, p = self._get_inputs(X)
        F = []
        for layer in self._layers:
            p = layer(X, p)
        F.append(p)

        return PrimitiveParameters.from_existing(F[-1], fit=F)


class SpacePartitionerPrimitiveNetwork(_BasePrimitiveNetwork):
    """
    Arguments
    ---------
        layers: list of nn.Module to predict the primitive parameters
        n_primitives: the depth of the binary tree that defines the number of
                      leaves and hence the maximum number of primitives
        feature_shape: the dimensions of the features provided by the feature
                       extractor
    """
    def __init__(self, layers_fit, layers_partition, n_primitives, feature_shape):
        super(SpacePartitionerPrimitiveNetwork, self).__init__()
        for i, l in enumerate(layers_fit):
            self.add_module("layer_fit{}".format(i), l)
        for i, l in enumerate(layers_partition):
            self.add_module("layer_partition{}".format(i), l)

        self._layers_fit = layers_fit
        self._layers_partition = layers_partition

        self._partitioning = nn.Sequential(
            nn.Linear(feature_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 2*feature_shape)
        )

        self._n_primitives = n_primitives

    def forward(self, X):
        F, p = self._get_inputs(X)

        # Initialize some variables
        B, D = F.shape

        # Produce the partitioning of the feature
        C = [F.unsqueeze(1)]
        for d in range(1, self._n_primitives):
            C.append(self._partitioning(C[d-1]).view(B, -1, D))

        # Predict primitives for each partition
        P = []
        for Ci in C:
            pcurrent = p
            for layer in self._layers_partition:
                pcurrent = layer(Ci, pcurrent)
            P.append(pcurrent)

        F = []
        for Ci in C:
            pcurrent = p
            for layer in self._layers_fit:
                pcurrent = layer(Ci, pcurrent)
            F.append(pcurrent)

        return PrimitiveParameters.from_existing(
            F[-1],
            space_partition=[C, P],
            fit=F
        )


class SingleDepthSpacePartitionerPrimitiveNetwork(_BasePrimitiveNetwork):
    """
    Arguments:
    ----------
        layers_fit: list of nn.Module to predict the primitive parameters
        layers_partition: list of nn.Modules to predict the space partitioning
        n_primitives: the depth of the binary tree that defines the number of
                      leaves and hence the maximum number of primitives
        feature_shape: the dimensions of the features provided by the feature
                       extractor
    """
    def __init__(
        self, layers_fit, layers_partition, n_primitives, feature_shape
    ):
        super(SingleDepthSpacePartitionerPrimitiveNetwork, self).__init__()
        for i, l in enumerate(layers_fit):
            self.add_module("layer_fit{}".format(i), l)
        for i, l in enumerate(layers_partition):
            self.add_module("layer_partition{}".format(i), l)

        self._layers_fit = layers_fit
        self._layers_partition = layers_partition
        self._n_primitives = n_primitives

    def forward(self, X):
        X, p = self._get_inputs(X)

        # Predict primitives for each partition
        P = []
        pcurrent = p
        for layer in self._layers_partition:
            pcurrent = layer(X, pcurrent)
        P.append(pcurrent)

        F = []
        pcurrent = p
        for layer in self._layers_fit:
            pcurrent = layer(X, pcurrent)
        F.append(pcurrent)

        return PrimitiveParameters.from_existing(
            F[-1],
            space_partition=[X, P],
            fit=F
        )


class ConstantSQ(nn.Module):
    def forward(self, X, primitive_params):
        B, M, _ = X.shape
        rotations = X.new_zeros(B, M*4)
        rotations[:, ::4] = 1.
        return PrimitiveParameters.from_existing(
            primitive_params,
            sizes=X.new_ones(B, M*3)*0.1,
            shapes=X.new_ones(B, M*2),
            rotations=rotations
        )


def get_primitive_network(network, feature_extractor, config):
    n_primitives = config["data"]["n_primitives"]
    layers = config["primitive_layer"]

    return dict(
        default=lambda: PrimitiveNetwork(
            get_layer_instances(layers, feature_extractor, n_primitives, config)
        ),
        hierarchical=lambda: HierarchicalPrimitiveNetwork(
            get_layer_instances(layers, feature_extractor, n_primitives, config),
            n_primitives,
            feature_extractor.feature_shape
        ),
        space_partitioner=lambda: SpacePartitionerPrimitiveNetwork(
            get_layer_instances(layers, feature_extractor, n_primitives, config),
            get_layer_instances(
                config["structure_layer"], feature_extractor, n_primitives, config
            ),
            n_primitives,
            feature_extractor.feature_shape
        ),
        single_space_partitioner=lambda: SingleDepthSpacePartitionerPrimitiveNetwork(
            get_layer_instances(layers, feature_extractor, n_primitives, config),
            get_layer_instances(
                config["structure_layer"], feature_extractor, n_primitives, config
            ),
            n_primitives,
            feature_extractor.feature_shape
        )
    )[network]()


def get_layer_instances(layers, feature_extractor, n_primitives, config):
    factories = {
        "probs": probs,
        "rotations": rotations,
        "shapes": shapes,
        "sizes": sizes,
        "translations": translations,
        "qos": qos,
        "sharpness": sharpness,
        "simple_constant_sq": simple_constant_sq,
        "constant": lambda *args: ConstantSQ()
    }
    layer_instances = []
    for name in layers:
        category, layer = name.split(":")
        layer_instances.append(factories[category](
            layer,
            feature_extractor,
            n_primitives,
            config
        ))

    return layer_instances
