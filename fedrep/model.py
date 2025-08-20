from base.model import ModelManager
from typing import Type, Optional, Dict, List, Union
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch import nn
import torch
from flwr.common import log
from logging import INFO

class ModelManagerFedrep(ModelManager):
    
    def __init__( self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        model_class:  Type[nn.Module],
        client_save_path: Optional[str] = ""):
        
        super().__init__(client_id, config, trainloader, testloader, model_class, client_save_path)
        self.local_body_epochs = self.config.client_config.local_body_epochs
        self.local_head_epochs = self.config.client_config.local_head_epochs
        
    def train(
        self,
        epochs: int = 1,
        verbose: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        
        if self.client_save_path is not None:
            try:
                self.model.local_net.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:   
                log(INFO, "No client state found, training from scratch.")
                pass
            
        criterion = nn.CrossEntropyLoss()
        weights = [v for k, v in self._model.named_parameters() if "weight" in k]
        biases = [v for k, v in self._model.named_parameters() if "bias" in k]
        
        optimizer = torch.optim.SGD(
             [
                {"params": weights, "weight_decay": self.config.client_config.weight_decay},
                {"params": biases, "weight_decay": 0.0},
            ], lr=self.config.client_config.learning_rate, momentum=self.config.client_config.momentum)
        correct, total = 0, 0
        epoch_loss = 0
        loss: torch.Tensor = 0.0
        self._model.train()
        self._model.disable_global_net()
        for _ in range(self.local_head_epochs):
            for batch in self.trainloader:
                optimizer.zero_grad()
                images, labels = batch['img'], batch['label']
                outputs = self._model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if verbose and _ >= epochs // 10 and _ % 10 == 0:
                log(INFO, f"Local Head Epoch {_+1}/{epochs}, Loss: {loss / len(self.trainloader):.4f}")
                
        self._model.enable_global_net()
        self._model.disable_local_net()
        for _ in range(self.local_body_epochs):
            for batch in self.trainloader:
                optimizer.zero_grad()
                images, labels = batch['img'], batch['label']
                outputs = self._model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if verbose and _ >= epochs // 10 and _ % 10 == 0:
                log(INFO, f"Local Body Epoch {_+1}/{epochs}, Loss: {loss / len(self.trainloader):.4f}")
                
        # Save client state (local_net)
        if self.client_save_path is not None:
            torch.save(self.model.local_net.state_dict(), self.client_save_path)

        return {"loss": epoch_loss / len(self.trainloader) / self.local_body_epochs / self.local_head_epochs, "accuracy": correct / total}
    