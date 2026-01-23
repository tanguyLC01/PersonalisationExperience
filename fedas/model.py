import torch.nn as nn
from collections import OrderedDict
import numpy as np
from typing import Any, Dict, List, Tuple, Type, Union, Optional, Protocol

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
import torch
from flwr.common import log
from logging import INFO
from sklearn.metrics import classification_report
from base.model import ModelSplit,  personalize_layer
from torch.autograd import grad
from base.model import ModelManager
from flwr.common import log
from logging import INFO

class ModelManagerFedas(ModelManager):
    """Manager for Fedas implementation"""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        model_class:  Type[nn.Module],
        client_save_path: Optional[str] = "",
    ):
        super().__init__(client_id, config, trainloader, testloader, model_class, client_save_path)
        self.model_class = model_class
        self.lr_align = self.config['client_config']['align_learning_rate']
        self.lr_local = self.config['client_config']['local_learning_rate']
        self.fim_trace_history = []
        self.class_accuracy_history = []  
        self.num_classes = self.config.dataset.num_classes 
        
    def _get_optimizer(self):
        weights = [v for k, v in self._model.named_parameters() if "weight" in k]
        biases = [v for k, v in self._model.named_parameters() if "bias" in k]
        optimizer = torch.optim.SGD(
             [
                {"params": weights, "weight_decay": self.config.client_config.weight_decay},
                {"params": biases, "weight_decay": 0.0},
            ], lr=self.config.client_config.local_learning_rate, momentum=self.config.client_config.momentum)
        return optimizer
    
    def compute_class_accuracy(self):
        correct = np.zeros(self.num_classes)
        total = np.zeros(self.num_classes)

        self._model.eval()

        with torch.no_grad():
            for batch in self.testloader:
                images, labels = batch['img'].to(self.device), batch['label'].to(self.device)
                outputs = self._model(images)
                preds = outputs.argmax(dim=1)

                for t, p in zip(labels, preds):
                    total[t.item()] += 1
                    if t == p:
                        correct[t.item()] += 1

        return [
            float(correct[c] / total[c]) if total[c] > 0 else 0.0
            for c in range(self.num_classes)
        ]
        
    def align(self, server_state_dict: Dict[str, torch.Tensor]) ->  Dict[str, torch.Tensor]:
        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]

        # print(f'client{id}')
        for batch in self.trainloader:
            x_batch = batch['img'].to(self.device)
            y_batch = batch['label'].to(self.device)

            with torch.no_grad():
                proto_batch = self.model.global_net(x_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        prototype_dim = proto_batch.shape[1:]
        mean_prototypes = torch.zeros(
            (self.num_classes, *prototype_dim), device=self.device
        )
        prototype_mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)

        for c, class_protos in enumerate(local_prototypes):
            if len(class_protos) > 0:
                stacked = torch.stack(class_protos).to(self.device)
                mean_prototypes[c] = stacked.mean(dim=0)
                prototype_mask[c] = True
                
        # Create server_model
        server_model = ModelSplit(self._create_model(self.model_class))
        personalize_layer(server_model, self.personalization_level)
        server_model.set_parameters(server_state_dict)  # type: ignore[attr-defined]
        server_model.to(self.device)
        server_model.train()

        alignment_optimizer = torch.optim.SGD(
            server_model.global_net.parameters(),
            lr=self.lr_align
        )
        alignment_loss_fn = torch.nn.MSELoss()
        
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for batch in self.trainloader:
                x_batch = batch['img'].to(self.device).float()
                y_batch = batch['label'].to(self.device)
                
                global_proto_batch = server_model.global_net(x_batch)
                
                valid_mask = prototype_mask[y_batch]
                if not valid_mask.any():
                    continue

                selected_outputs = global_proto_batch[valid_mask]
                selected_targets = mean_prototypes[y_batch[valid_mask]]

                loss = alignment_loss_fn(
                    selected_outputs,
                    selected_targets
                )

                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()
                
        return server_model.state_dict()
    
    def get_alpha(self) -> float:
        self._model.eval()
        self._model.to(self.device)
        # print(f'client{self.id}, start cal fim.')
        # Compute FIM and its trace after training
        fim_trace_sum = 0
        for batch in self.trainloader:
            x, y = batch['img'], batch['label']
            # Forward pass
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x)
            # Negative log likelihood as our loss
            nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

            # Compute gradient of the negative log likelihood w.r.t. model parameters
            grads = grad(nll, self.model.parameters())

            # Compute and accumulate the trace of the Fisher Information Matrix
            for g in grads:
                fim_trace_sum += torch.sum(g ** 2).detach()

        # add the fisher log
        self.fim_trace_history.append(fim_trace_sum.item())
        
        class_acc = self.compute_class_accuracy()
        self.class_accuracy_history.append(class_acc)
        return fim_trace_sum.item()
