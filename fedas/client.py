from flwr.client import NumPyClient
from typing import List
from base.client import PersonalizedClient
import torch
from collections import OrderedDict
import numpy as np
from typing import Tuple, List, Dict, Union
from flwr.common import NDArrays, Scalar
from flwr.common import Context
from omegaconf import DictConfig
import os
from base.model import ModelManager 
from flwr.common.logger import log
from logging import INFO

class FedasClient(PersonalizedClient):
    
    def __init__(self, partition_id, model_manager: ModelManager, config: DictConfig):
        super().__init__(partition_id, model_manager, config)
    
    def set_parameters(self, parameters, evaluate = False):
        # Retransform parameters 
        
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_global_net") or k.startswith("_local_net")
        ]
        params_dict = zip(model_keys, parameters)

        server_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # Alignement
        aligned_model_state_dict = self.model_manager.align(server_state_dict)
        self.model_manager.model.set_parameters(aligned_model_state_dict) # type: ignore[attr-defined]     
    
    def fit(self, parameters, config):
        local_model_param, train_dataset_size, metrics = super().fit(parameters, config) # On renvoie theta ou alors \delta theta 
        metrics['alpha'] = self.model_manager.get_alpha()
        return local_model_param, train_dataset_size, metrics