from base.model import ModelManager
from typing import Dict, Type, Optional
from torch import nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.common import Scalar
from flwr.common import log
from logging import INFO


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

    def finetune_model(self) -> None:
        
        # For the test of the Exploiting Share Representation, we only fine-tune on the head
        self._model.disable_global_net()
        self._model.enable_local_net()
        for _ in range(self.config.client_config.finetuned_epoch):
            self.train(epochs=1)
            loss, accuracy = self.test().values()
            log(INFO, f'Epoch {_+1}/{self.config.client_config.finetuned_epoch}, Loss: {loss}", Accuracy : {accuracy:.4f})')
            
