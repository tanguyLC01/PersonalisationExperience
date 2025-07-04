from flwr.client import ClientApp
from flwr.common import Context

from flwr.simulation import run_simulation
from flwr.server import ServerAppComponents
from logging import Logger
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms 
from flwr_datasets import FederatedDataset
from typing import Callable, Tuple
from omegaconf import DictConfig

from fedper.client import PersonalizedClient, BaseClient
from flwr.server import ServerConfig, ServerAppComponents
from fedper.server import PartialLayerFedAvg, weighted_average

from fedper.mobile_model import MobileNetModelManager
import logging

log = logging.getLogger(__name__)
import os

def get_server_fn(cfg: DictConfig, server_path: str) -> Callable[[Context], ServerAppComponents]:
    
    
    def fit_config(server_round: int) -> dict:
        return {
            "batch_size": cfg.client_config.batch_size,
            "local_epochs": cfg.client_config.num_epochs,
            "optimizer": cfg.client_config.optimiser,  # Use sgd optimizer'
            "learning_rate": cfg.client_config.learning_rate,  # Learning rate for the optimizer
        }
    
    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.

        You can use settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """

        # Create FedAvg strategy
        strategy = PartialLayerFedAvg(
            save_path=server_path,
            on_fit_config_fn=fit_config,
            fraction_fit=cfg.server_config.fraction_fit,
            fraction_evaluate=cfg.server_config.fraction_evaluate,
            min_fit_clients=cfg.server_config.min_fit_clients,
            min_evaluate_clients=cfg.server_config.min_evaluate_clients,
            min_available_clients=cfg.server_config.min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average
        )

        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=cfg.num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)
    
    return server_fn

def load_datasets(partition_id: int, fds: FederatedDataset, cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    partition = fds.load_partition(partition_id, split='train')
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=cfg.seed)   
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    def apply_transforms(batch: dict) -> dict:
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=cfg.client_config.batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=cfg.client_config.batch_size)
    testset = fds.load_partition(partition_id, split='test').with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.client_config.batch_size)
        
    return trainloader, valloader, testloader
    


def get_client_fn(cfg: DictConfig, client_save_path: str, fds: FederatedDataset) -> Callable[[Context], BaseClient]:

    def client_fn(context: Context) -> BaseClient:
        partition_id = context.node_config['partition-id']
        client_local_net_model_path = f"{client_save_path}/local_net_{partition_id}.pth"
        trainloader, valloader, _ = load_datasets(partition_id, fds, cfg)
        mobile_net_manager = MobileNetModelManager(partition_id, cfg, trainloader, valloader, client_local_net_model_path)
        return PersonalizedClient(partition_id, mobile_net_manager, cfg).to_client()
    
    return client_fn    

