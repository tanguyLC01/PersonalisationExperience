from base.model import ModelManager
from typing import Dict, Type, Optional
from torch import nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.common import Scalar
class ModelManagerFedavgft(ModelManager):
    
    def __init__(self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        model_class:  Type[nn.Module],
        client_save_path: Optional[str] = "") -> None:
        
        super().__init__(client_id, config,
        trainloader,
        testloader,
        model_class,
        client_save_path)
    
    def finetune_model(self, finetuned_epochs: int) -> None:
        self._model.disable_global_net()
        self.train(epochs=finetuned_epochs)