import torch
import torch.nn as nn
import torch.optim as optim

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from .feature_extractors import get_feature_extractor
from .primitive_layer import get_primitive_network
from .utils import FrozenBatchNorm2d
from ..utils.value_registry import ValueRegistry


class Network(nn.Module):
    """A module used to represent the general network architecture, which
    consists of a features extractor and a primitive layer. The
    features_extractor takes an input (it can be anything) and estimates a set
    of features. The primitive_layer, encodes these features to Mx13
    parameters, which correspond to the M primitives.
    """
    def __init__(self, feature_extractor, primitive_layer):
        super(Network, self).__init__()
        self._feature_extractor = feature_extractor
        self._primitive_layer = primitive_layer

    def forward(self, X):
        return self._primitive_layer(self._feature_extractor(X))


class NetworkBuilder(object):
    def __init__(self, config):
        self._config = config

    @classmethod
    def from_yaml_file(cls, filepath):
        with open(filepath, "r") as f:
            config = yaml.load(f, Loader=Loader)
        return cls(config)

    @property
    def config(self):
        return self._config

    @property
    def feature_extractor(self):
        return get_feature_extractor(
            self.config["feature_extractor"]["type"],
            self.config["data"]["n_primitives"],
            self.config["feature_extractor"].get("freeze_bn", False)
        )

    @property
    def primitive_layer(self):
        return get_primitive_network(
            self.config.get("primitive_network", "default"),
            self.feature_extractor,
            self.config
        )

    @property
    def network(self):
        return Network(self.feature_extractor, self.primitive_layer)


def train_on_batch(
    network,
    optimizer,
    loss_fn,
    X,
    y_target,
    current_epoch
):
    """Perform a forward and backward pass on a batch of samples and compute
    the loss and the primitive parameters.
    """
    training_stats = ValueRegistry.get_instance("training_stats")
    training_stats["current_epoch"] = current_epoch
    optimizer.zero_grad()
    # Do the forward pass to predict the primitive_parameters
    y_hat = network(X)
    loss = loss_fn(y_hat, y_target)
    # Do the backpropagation
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 1)
    # Do the update
    optimizer.step()

    return (
        loss.item(),
        [x.data if hasattr(x, "data") else x for x in y_hat],
    )


def validate_on_batch(
    network,
    loss_fn,
    X,
    y_target
):
    """Perform a forward pass on a batch of samples and compute
    the loss and the metrics.
    """
    # Do the forward pass to predict the primitive_parameters
    y_hat = network(X)
    loss = loss_fn(y_hat, y_target)
    return (
        loss.item(),
        [x.data if hasattr(x, "data") else x for x in y_hat],
    )


def optimizer_factory(config, model):
    """Based on the input arguments create a suitable optimizer object
    """
    params = model.parameters()

    optimizer = config["loss"].get("optimizer", "Adam")
    lr = config["loss"].get("lr", 1e-3)
    momentum = config["loss"].get("momentum", 0.9)

    if optimizer == "SGD":
        return optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        return optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError()


def build_network(config_file, weight_file, device="cpu"):
    network = NetworkBuilder.from_yaml_file(config_file).network
    # Move the network architecture to the device to be used
    network.to(device)
    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )
    return network
