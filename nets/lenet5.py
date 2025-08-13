from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

class Lenet5(nn.Module):
    def __init__(self, model_config):
        super(Lenet5, self).__init__()
        num_classes = model_config['num_classes']    
        in_channels = model_config['in_channels'] 
        self.global_net = nn.Sequential(
            OrderedDict(
                [
                ('conv1', nn.Sequential(
                    nn.Conv2d(in_channels, 6, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )),
                ('conv2', nn.Sequential(
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Sequential(
                    nn.Linear(16 * 5 * 5, 120),
                    nn.ReLU()
                )),
                ('fc2', nn.Sequential(
                    nn.Linear(120, 84),
                    nn.ReLU()
                )),
                ('fc3', nn.Linear(84, num_classes))
                ]
            )
        )
        
        self.local_net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.global_net(x)
        return self.local_net(x)
