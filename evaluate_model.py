import torch
import pickle
import os
from typing import Dict, Any
from fedper.mobile_model import MobileNetModelManager
from PersonalisationExperience.base.model import ModelManager
from PersonalisationExperience.base.partitioner import VariablePathologicalPartitioner
import argparse
from omegaconf import OmegaConf
from PersonalisationExperience.base.utils import load_datasets
from flwr_datasets import FederatedDataset
import json
import numpy as np

def load_global_weights(global_weights_path: str, net_manager: ModelManager) -> None:
    with open(global_weights_path, 'rb') as f:
        data = pickle.load(f)
            
        ndarrays = data['global_parameters']

        state_dict = net_manager.model.global_net.state_dict()

        new_state_dict = {}
        for key, array in zip(state_dict.keys(), ndarrays):
            new_state_dict[key] = torch.tensor(array, dtype=torch.float32)
        
        net_manager.model.global_net.load_state_dict(new_state_dict)
                
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Personalized Models")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    parser.add_argument('-n', '--num_client', type=int, default=0, help='Client Id to evalute')
    log_directory  = parser.parse_args().log_directory
    client_id = parser.parse_args().num_client
    config_file = f"{log_directory}/.hydra/config.yaml"
    cfg = OmegaConf.load(config_file)
    
    client_local_net_model_path = f'{log_directory}/client_states/local_net_{client_id}.pth'

    if cfg.dataset.partitioner.name == 'variable_pathological':
        partitioner_train = VariablePathologicalPartitioner(cfg.num_clients, partition_by='label', num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition, seed=cfg.seed, shuffle=True)
        partitioner_test = VariablePathologicalPartitioner(cfg.num_clients, partition_by='label', num_classes_per_partition=cfg.dataset.partitioner.num_classes_per_partition, seed=cfg.seed, shuffle=True)
    
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={'train': partitioner_train, "test": partitioner_test})    
    trainloader, _, testloader = load_datasets(client_id, fds, cfg)
    
    train_targets = []
    for batch in trainloader:
        train_targets.extend(batch['label'].cpu().numpy())
    train_targets = np.array(train_targets)
    unique_test, counts_test = np.unique(train_targets, return_counts=True)
    print("Support (number of samples) for each class in the trainq split (from trainloader):")
    for cls, count in zip(unique_test, counts_test):
        print(f"Class {cls}: {count} samples")
   
    mobile_net_manager = MobileNetModelManager(client_id, cfg, trainloader, testloader, client_local_net_model_path)
    # We set the global_parameters as there is no server for testing. The managers handles the local net part on its own.
    if os.path.exists(f'{log_directory}/server_state'):
        load_global_weights(f'{log_directory}/server_state/parameters_round_{cfg.num_rounds}.pkl', mobile_net_manager)
    else: # It means the model is fully localised
        mobile_net_manager.client_save_path = None
        state_dict = torch.load(f'{log_directory}/client_states/local_net_{client_id}.pth')
        mobile_net_manager.model.load_state_dict(state_dict)
        
    res_dict = mobile_net_manager.test(full_report=True)

    # Save metrics to log file
    metrics = {
        "loss": res_dict['loss'],
        "classification_report": res_dict['report'],
        "accuracy": res_dict['accuracy']
    }
    metrics_path = f"{log_directory}/test_metrics_{client_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Per-class metrics:")
    for cls, metrics in res_dict['report'].items():
        if cls.isdigit():
            print(f"Class {cls} ({[int(cls)]}): Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")
    
if __name__ == "__main__":
    main()
