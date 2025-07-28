from flwr.client import ClientApp
from flwr.simulation import run_simulation
import numpy as np
import random
from flwr_datasets import FederatedDataset
import os
import hydra
from omegaconf import DictConfig
import importlib
from flwr.server import ServerApp
from base.utils import get_server_fn, get_client_fn
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions
from base.partitioner import load_partitioner
from flwr.common import Context
from flwr.common import RecordDict
from logging import INFO
from flwr.common import log

#partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by="label",
#                                   alpha=0.1, min_partition_size=10)


@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
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
   
        
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": train_partitioner, "test":  test_partitioner})
    

    fig_train, _, _ = plot_label_distributions(partitioner=fds.partitioners["train"],
    label_name="label",
    legend=True,
    )
    
    fig_test, _, _ = plot_label_distributions(partitioner=fds.partitioners["test"],     
                                              label_name="label",
                                                legend=True,
                                                )
                                                                        
    fig_test.legend()
    fig_test.set_size_inches(12, 8)
    fig_train.legend()
    fig_train.set_size_inches(12, 8)

    fig_train.savefig(f'{log_save_path}/samples_per_label_per_client.png', dpi=300)
    fig_test.savefig(f'{log_save_path}/samples_per_label_per_client_test.png', dpi=300)
    plt.close()

    
    # Create a new client
    log(INFO, f"ModelManager path : {cfg.algorithm}.model.ModelManager{cfg.algorithm.capitalize()}")
    try:
        model_manager = getattr(importlib.import_module(f'{cfg.algorithm}.model'), f'ModelManager{cfg.algorithm.capitalize()}')
    except ModuleNotFoundError:
        model_manager = getattr(importlib.import_module(f'base.model'), 'ModelManager')  
    log(INFO, f"Client will use {model_manager} as model manager class")
    model_name = ''.join(word.capitalize() for word in cfg.model.model_class_name.split('_')) # If a model file is mobile_net, the model name is MobileNet
    model_module = getattr(importlib.import_module(f'nets.{cfg.model.model_class_name}'), model_name)
    client_class_name = getattr(importlib.import_module(f'base.client'), cfg.client_config.client_class_name)
    client_fn = get_client_fn(cfg, client_save_path, fds, model_manager, model_module, client_class_name)
    client = ClientApp(client_fn)
    print(client)
    
    # Create a new server
    server_fn = get_server_fn(cfg, server_save_path)
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    history = run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": 
            {"num_cpus": cfg.client_config.num_cpus, 
             "num_gpus": cfg.client_config.num_gpus}
            }
    )
    
    
    if cfg.algorithm == 'fedavg-ft':
        for client_id in range(cfg.num_clients):
            client = client_class_name(client_id, model_manager, cfg)
            client.model_manager.finetune_model(cfg.client_config.finetuned_epochs)

    print(history)

if __name__ == "__main__":
    main()