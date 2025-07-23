
import torch
import torch.nn as nn
from base.model import ModelSplit
from typing import Tuple, Dict

# Set model architecture
ARCHITECTURE = {
    "layer_1": {"conv_dw": [32, 64, 1]},
    "layer_2": {"conv_dw": [64, 128, 2]},
    "layer_3": {"conv_dw": [128, 128, 1]},
    "layer_4": {"conv_dw": [128, 256, 2]},
    "layer_5": {"conv_dw": [256, 256, 1]},
    "layer_6": {"conv_dw": [256, 512, 2]},
    "layer_7": {"conv_dw": [512, 512, 1]},
    "layer_8": {"conv_dw": [512, 512, 1]},
    "layer_9": {"conv_dw": [512, 512, 1]},
    "layer_10": {"conv_dw": [512, 512, 1]},
    "layer_11": {"conv_dw": [512, 512, 1]},
    "layer_12": {"conv_dw": [512, 1024, 2]},
    "layer_13": {"conv_dw": [1024, 1024, 1]},
}


class MobileNet(nn.Module):
    """Model from MobileNet-v1 (https://github.com/wjc852456/pytorch-mobilenet-v1)."""

    def __init__(
        self,
        model_config: Dict
    ) -> None:
        num_classes = model_config['num_classes']
        in_channels  = model_config['in_channels']
        super(MobileNet, self).__init__()

        self.architecture = ARCHITECTURE

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.global_net = nn.Sequential()
        self.global_net.add_module("initial_batch_norm", conv_bn(in_channels, 32, 2))
        for i in range(1, 13):
            for _, value in self.architecture[f"layer_{i}"].items():
                self.global_net.add_module(f"conv_dw_{i}", conv_dw(*value))

        self.global_net.add_module('classification_module', nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1))
                                                                          , nn.Flatten()
                                                                          ,nn.Linear(1024, num_classes)))
        self.local_net = nn.Identity()

            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.global_net(x)
        return self.local_net(x)