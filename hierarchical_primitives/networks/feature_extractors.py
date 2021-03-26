import torch
import torch.nn as nn
from torchvision import models

from .primitive_parameters import PrimitiveParameters
from .utils import FrozenBatchNorm2d


class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks.
    """
    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, X):
        return self._feature_extractor(X)


class TulsianiFeatures(BaseFeatureExtractor):
    """Build a variation of the feature extractor implemented in the volumetric
    primitives paper of Shubham Tulsiani.

    https://arxiv.org/pdf/1612.00404.pdf
    """
    def __init__(self, freeze_bn):
        super(TulsianiFeatures, self).__init__()

        # Declare and initiliaze some useful variables
        n_filters = 4
        input_channels = 1
        encoder_layers = []
        # Create an encoder using a stack of convolutions
        for i in range(5):
            encoder_layers.append(
                nn.Conv3d(input_channels, n_filters, kernel_size=3, padding=1)
            )
            if not freeze_bn:
                encoder_layers.append(nn.BatchNorm3d(n_filters))
            encoder_layers.append(nn.LeakyReLU(0.2, True))
            encoder_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

            input_channels = n_filters
            # Double the number of filters after every layer
            n_filters *= 2

        # Add the two fully connected layers
        input_channels = n_filters / 2
        n_filters = 100
        for i in range(2):
            encoder_layers.append(nn.Conv3d(input_channels, n_filters, 1))
            #encoder_layers.append(nn.BatchNorm3d(n_filters))
            encoder_layers.append(nn.LeakyReLU(0.2, True))

            input_channels = n_filters

        self._feature_extractor = nn.Sequential(*encoder_layers[:-1])
        self._feature_shape = n_filters

    def forward(self, X):
        return self._feature_extractor(X).view(X.shape[0], -1)


class ResNet18(BaseFeatureExtractor):
    """Build a feature extractor using the pretrained ResNet18 architecture for
    image based inputs.
    """
    def __init__(self, freeze_bn):
        super(ResNet18, self).__init__()
        self._feature_extractor = models.resnet18(pretrained=True)
        if freeze_bn:
            FrozenBatchNorm2d.freeze(self._feature_extractor)

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._feature_shape = 512


def get_feature_extractor(name, n_primitives, freeze_bn=False):
    """Based on the name return the appropriate feature extractor"""
    return {
        "tulsiani": lambda: TulsianiFeatures(freeze_bn),
        "resnet18": lambda: ResNet18(freeze_bn)
    }[name]()
