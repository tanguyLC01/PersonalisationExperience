from flwr.client import NumPyClient
from typing import List
import torch
from collections import OrderedDict
import numpy as np
from fedper.model import train, test
from typing import Tuple, List
from flwr.common import Context
import os


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerNumPyClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, client_save_path) -> None:
        self.client_local_model_path = f"{client_save_path}/local_net_{partition_id}.pth"
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return the current parameters of the global network."""
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net.global_net)

    def fit(self, parameters, config) -> List[np.ndarray]:
        """Train the local network and return the updated parameters."""
        print(f"[Client {self.partition_id}] fit, config: {config}")
        if os.path.exists(self.client_local_model_path):
            self.net.local_net.load_state_dict(torch.load(self.client_local_model_path))
        set_parameters(self.net.global_net, parameters)
        train(self.net, self.trainloader, epochs=4)
        torch.save(self.net.local_net.state_dict(), self.client_local_model_path)
        return get_parameters(self.net.global_net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        """Evaluate the local network and return the loss and accuracy."""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        if os.path.exists(self.client_local_model_path):
            self.net.local_net.load_state_dict(torch.load(self.client_local_model_path))
        set_parameters(self.net.global_net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
    
