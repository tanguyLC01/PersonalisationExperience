import os
import argparse
from omegaconf import OmegaConf
from base.utils import load_datasets
from flwr_datasets import FederatedDataset
import json
import numpy as np
from base.partitioner import load_partitioner
from load_classname import load_client_element
from base.utils import load_global_weights
                
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Personalized Models")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    parser.add_argument('-n', '--num_client', type=int, default=0, help='Client Id to evalute')
    log_directory  = parser.parse_args().log_directory
    client_id = parser.parse_args().num_client
    config_file = f"{log_directory}/.hydra/config.yaml"
    cfg = OmegaConf.load(config_file)
    
    client_local_net_model_path = f'{log_directory}/client_states/local_net_{client_id}.pth'

    partitioner_train, partitioner_test = load_partitioner(cfg)
    
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
     
    _, model_manager, model_module = load_client_element(cfg)
    net_manager = model_manager(client_id, cfg, trainloader, testloader, model_class=model_module, client_save_path=client_local_net_model_path)

    # We set the global_parameters as there is no server for testing. The managers handles the local net part on its own.
    if os.path.exists(f'{log_directory}/server_state'):
        load_global_weights(f'{log_directory}/server_state/parameters_round_{cfg.num_rounds}.pkl', net_manager)
        
    res_dict = net_manager.test(full_report=True)

    # Save metrics to log file
    metrics = {
        "loss": res_dict['loss'],
        "classification_report": res_dict['report'],
        "accuracy": res_dict['accuracy']
    }
    os.makedirs(f"{log_directory}/test_metrics", exist_ok=True)
    metrics_path = f"{log_directory}/test_metrics/test_metrics_{client_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Per-class metrics:")
    for cls, metrics in res_dict['report'].items():
        if cls.isdigit():
            print(f"Class {cls} ({[int(cls)]}): Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")
    
if __name__ == "__main__":
    main()
