from torch import nn
import torch.nn.functional as F

class CnnNet(nn.Module):
    def __init__(self, model_config):
        super(CnnNet, self).__init__()
        num_classes = model_config['num_classes']
        self.global_net = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
        self.local_net = nn.Identity()

    def forward(self, x):
        x = self.global_net(x)
        return self.local_net(x)