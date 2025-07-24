import torch.nn as nn
from collections import OrderedDict
import numpy as np
from typing import Any, Dict, List, Tuple, Type, Union, Optional

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
import torch
from flwr.common import log
from logging import INFO
from sklearn.metrics import classification_report




class ModelSplit(nn.Module):
    """Class for splitting a model into global_net and local_net."""

    def __init__(
        self,
        model: nn.Module,
    ):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
        """
        super().__init__()

        self._global_net, self._local_net = self._get_model_parts(model)


    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the global_net and local_net of the model.

        Args:
            model: model to be split into local_net and global_net

        Returns
        -------
            Tuple where the first element is the global_net of the model
            and the second is the local_net.
        """
        return model.global_net, model.local_net

    @property
    def global_net(self) -> nn.Module:
        """Return model global_net."""
        return self._global_net

    @global_net.setter
    def global_net(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model global_net.

        Args:
            state_dict: dictionary of the state to set the model global_net to.
        """
        self.global_net.load_state_dict(state_dict, strict=True)

    @property
    def local_net(self) -> nn.Module:
        """Return model local_net."""
        return self._local_net

    @local_net.setter
    def local_net(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model local_net.

        Args:
            state_dict: dictionary of the state to set the model local_net to.
        """
        self.local_net.load_state_dict(state_dict, strict=True)
        
    def get_global_net_children_name(self) -> List[str]:
        children = list(self.global_net.named_children())
        return list(map(lambda x: x[0], children))

    def get_parameters(self) -> List[np.ndarray]:
        """Get global_net parameters 

        Returns
        -------
            global_net parameters
        """
        return [
            val.cpu().numpy()
            for val in [
                *self.global_net.state_dict().values()
            ]
        ]
        
    def personalise_last_module(self):
        last_child = list(self.global_net.children())[-1]
        self._local_net = nn.Sequential(last_child, *(list(self.local_net.children())))
        self._global_net = nn.Sequential(*(list(self.global_net.children())[:-1]))

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update(dict(state_dict.items()))
        self.load_state_dict(ordered_state_dict, strict=False)

    def enable_local_net(self) -> None:
        """Enable gradient tracking for the local_net parameters."""
        for param in self.local_net.parameters():
            param.requires_grad = True

    def enable_global_net(self) -> None:
        """Enable gradient tracking for the global_net parameters."""
        for param in self.global_net.parameters():
            param.requires_grad = True

    def disable_local_net(self) -> None:
        """Disable gradient tracking for the local_net parameters."""
        for param in self.local_net.parameters():
            param.requires_grad = False

    def disable_global_net(self) -> None:
        """Disable gradient tracking for the global_net parameters."""
        for param in self.global_net.parameters():
            param.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the global_net and the local_net."""
        x = self.global_net(inputs)
        return self.local_net(x)
    

class ModelManager:
    """Manager for models with global_net/local_net split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        model_class:  Type[nn.Module],
        client_save_path: Optional[str] = "",
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into global_net and local_net\
                (concrete implementation of ModelSplit).
        """
        super().__init__()
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_save_path = client_save_path
        log(INFO, f'Client save path : {self.client_save_path}')
        self.client_id = client_id
        self.config = config
        self.device = self.config.device
        self._model = ModelSplit(self._create_model(model_class))
        
        for _ in range(config.model.personalisation_level[client_id]):
            self._model.personalise_last_module()

    def _create_model(self, model_class) -> nn.Module:
        """Return model to be splitted into local_net and global_net."""
        return model_class(self.config.model).to(self.device)


    def train(
        self,
        epochs: int = 1,
        verbose: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """
        
        if self.client_save_path is not None:
            try:
                self.model.local_net.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:   
                log(INFO, "No client state found, training from scratch.")
                pass
            
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.client_config.learning_rate)
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        self.model.train()
        for _ in range(epochs):
            for batch in self.trainloader:
                optimizer.zero_grad()
                images, labels = batch['img'], batch['label']
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if verbose and _ >= epochs // 10 and _ % 10 == 0:
                log(INFO, f"Epoch {_+1}/{epochs}, Loss: {loss / len(self.trainloader):.4f}")
                
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
            try:
                self.model.local_net.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:   
                log(INFO, "No client state found, evaluating from scratch")
                pass
        
        self.model.eval()
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
    

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model

        
