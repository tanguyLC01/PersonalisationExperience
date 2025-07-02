"""MobileNet-v1 model, model manager and model split."""

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedper.model import ModelManager, ModelSplit
from flwr.common.logger import log
from logging import INFO
from sklearn.metrics import classification_report

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
        num_local_net_layers: int = 1,
        num_classes: int = 10,
        in_channels: int = 3, 
    ) -> None:
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

        self.global_net.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.global_net.add_module("flatten", nn.Flatten())
        self.global_net.add_module("fc", nn.Linear(1024, num_classes))
        self.local_net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.global_net(x)
        return self.local_net(x)


class MobileNetModelSplit(ModelSplit):
    """Split MobileNet model into global_net and local_net."""

    def _get_model_parts(self, model: MobileNet) -> Tuple[nn.Module, nn.Module]:
        return model.global_net, model.local_net


class MobileNetModelManager(ModelManager):
    """Manager for models with global_net/local_net split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_save_path: Optional[str] = ""    
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
        """
        super().__init__(
            model_split_class=MobileNetModelSplit,
            client_id=client_id,
            config=config,
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config.device
        self.client_save_path = client_save_path if client_save_path != "" else None
        self.learning_rate = self.config.client_config.learning_rate
        self.client_feats = None
        self.batch_size = self.config.client_config.batch_size

        

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into local_net and global_net."""
        try:
            return MobileNet(
                num_local_net_layers=self.config["model"]["personalisation_level"],
                num_classes=self.config["model"]["num_classes"],
                in_channels=self.config["model"]["in_channels"]
            ).to(self.device)
        except AttributeError:
            self.device = self.config.device
            return MobileNet(
                num_local_net_layers=self.config["model"]["personalisation_level"],
                num_classes=self.config["model"]["num_classes"],
                in_channels=self.config["model"]["in_channels"]
            ).to(self.device)

    def train(
        self,
        epochs: int = 1
        ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from simple MobileNet-v1 (PyTorch) \
        https://github.com/wjc852456/pytorch-mobilenet-v1.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (local_net) if client_save_path is not None and it is not empty
        if self.client_save_path is not None:
            try:
                self.model.local_net.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:   
                print("No client state found, training from scratch.")
                pass
            
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        # self.model.train()
        for _ in range(epochs):
            for step, batch in enumerate(self.trainloader):
                optimizer.zero_grad()
                images, labels = batch['img'], batch['label']
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # Save client state (local_net)
        if self.client_save_path is not None:
            torch.save(self.model.local_net.state_dict(), self.client_save_path)
        

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(
        self, 
        full_report: bool = False
    ) -> Dict[str, Any]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """
        # Load client state (local_net)
        if self.client_save_path is not None:
            self.model.local_net.load_state_dict(torch.load(self.client_save_path))

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        if full_report:
            all_preds = []
            all_targets = []
        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch['img'], batch['label']
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                predicted = torch.max(outputs.data, 1)[1]
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if full_report:
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    
        final_dict = {
            "loss": loss / len(self.testloader.dataset),
            "accuracy": correct / total,
        }
        
        if full_report:    
            final_dict['report'] = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
            
        print("Test Accuracy: {:.4f}".format(correct / total))

        return final_dict
        

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader) + len(self.testloader)
    
        
