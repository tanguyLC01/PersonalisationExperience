from flwr.client import ClientApp
from flwr.simulation import run_simulation
import numpy as np
import random
from flwr_datasets import FederatedDataset
import os
import hydra
from omegaconf import DictConfig
from flwr.server import ServerApp
from base.utils import get_server_fn, get_client_fn
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions
from base.partitioner import load_partitioner
from load_classname import load_client_element
from logging import INFO
from flwr.common import log
import torch
import logging
from base.dataset import BDD100KDataset
from base.utils import load_datasets, load_global_weights
from base.dataset import load_dbb100k_dataset

main_logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    log_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log(INFO, f"Saving logs to {log_save_path}")
    client_save_path = (
            f"{log_save_path}/client_states"
        )
    server_save_path = (
            f"{log_save_path}/server_state"
    )
    os.makedirs(client_save_path)
    os.makedirs(server_save_path)
    
    # --------------------- Choose the right Partitioner (Train and test will have the same) ---------------------
    train_partitioner, test_partitioner = load_partitioner(cfg)

    fds = load_dbb100k_dataset(cfg.dataset.root_path)
    train_partitioner.dataset = fds['train']
    test_partitioner.dataset = fds['test']
    
    
    # Create a new client
    client_class_name, model_manager, model_module = load_client_element(cfg)
    
    client_fn = get_client_fn(cfg, client_save_path, {'train': train_partitioner, 'test': test_partitioner}, model_manager, model_module, client_class_name)
    client = ClientApp(client_fn)
    print(client)
    
    # Create a new server
    server_fn = get_server_fn(cfg, server_save_path)
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": 
            {"num_cpus": cfg.client_config.num_cpus, 
             "num_gpus": cfg.client_config.num_gpus}
            }
    )
        
    if cfg.algorithm == 'fedavgft':
        for client_id in range(cfg.num_clients):
            client_local_net_model_path = f"{client_save_path}/local_net_{client_id}.pth"
            trainloader, _, testloader = load_datasets(client_id, fds, cfg)
            mobile_net_manager = model_manager(client_id, cfg, trainloader, testloader, model_class=model_module, client_save_path=client_local_net_model_path)
            client = client_class_name(client_id, mobile_net_manager, cfg)
            # Set the global weights of the model
            load_global_weights(os.path.join(server_save_path, f'parameters_round_{cfg.num_rounds}.pkl'), client.model_manager)   
            log(INFO, f"Fine-tuning client {client_id}")
            client.model_manager.finetune_model() 
            


if __name__ == "__main__":
    main()