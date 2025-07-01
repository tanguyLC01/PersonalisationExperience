import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flwr.client import ClientApp
from flwr.simulation import run_simulation
import numpy as np
import random
from flwr_datasets import FederatedDataset
import os
import hydra
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from fedper.partitioner import DirichletSkewedPartitioner, VariablePathologicalPartitioner
from flwr.server import ServerApp
from fedper.utils import load_datasets
from fedper.mobile_model import MobileNet
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions


log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed) 

    log_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Saving logs to {log_save_path}")
    client_save_path = (
            f"{log_save_path}/client_states"
        )
    
    os.makedirs(client_save_path)
    
    if cfg.dataset.partitioner.name == "dirichlet":
        partitioner = DirichletPartitioner(alpha=cfg.dataset.partitioner.alpha, num_partitions=cfg.num_clients, partition_by="label", seed=cfg.seed)
    elif cfg.dataset.partitioner.name == "dirichletskew":
        partitioner = DirichletSkewedPartitioner(num_partitions=cfg.num_clients, rich_clients=[0], alpha_rich=cfg.dataset.partitioner.alpha_rich,  alpha_poor=cfg.dataset.partitioner.alpha_poor, seed=cfg.seed)
    elif cfg.dataset.partitioner.name == "variable_pathological":
        partitioner = VariablePathologicalPartitioner(
            num_partitions=cfg.num_clients,
            partition_by="label",
            num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
            shuffle=True,
            seed=cfg.seed,
        )
        
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": partitioner, "test":  VariablePathologicalPartitioner(
        num_partitions=cfg.num_clients,
        partition_by="label",
        num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition,
        shuffle=True,
        seed=cfg.seed,
    )})
    
    epochs = cfg.client_config.num_epochs * cfg.num_rounds
    for client_id in range(cfg.num_clients):
        log.info(f"------------------ Training Client {client_id} ------------------")
        trainloader, valloader, _ = load_datasets(client_id, fds, cfg)
        model = MobileNet(0, cfg.model.num_classes).to(cfg.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.client_config.learning_rate)
        correct, total = 0, 0
        
        # self.model.train()
           
        train_losses = []  
        val_accuracies = []
        for _ in range(epochs): 
            
            model.train()
            running_loss = 0
            for batch in trainloader:
                optimizer.zero_grad()
                images, labels = batch['img'], batch['label']
                outputs = model(images.to(cfg.device))
                labels = labels.to(cfg.device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(trainloader)
            train_losses.append(avg_train_loss)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in valloader:
                    images, labels = batch['img'], batch['label']
                    outputs = model(images.to(cfg.device))
                    labels = labels.to(cfg.device)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                    total += labels.size(0)
            
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            if _ % 10 == 0 and _ >= 10:
                log.info(f"Epoch {_+1}/{epochs}, Loss: {avg_train_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")
        
        # Plot Loss and Accuracy
        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        
        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), val_accuracies, marker='s', linestyle='-', color='r', label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy Over Epochs')
        plt.legend()
        
        plt.savefig(f'{log_save_path}/train_loss_val_accuracy_{client_id}')
        
        torch.save(model.state_dict(), f'{client_save_path}/local_net_{client_id}')
    
    
if __name__ == "__main__":
    main()

        
        

    