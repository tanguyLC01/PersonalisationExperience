from flwr.client import ClientApp
from flwr.simulation import run_simulation
import numpy as np
import random
from flwr_datasets import FederatedDataset
import os
import hydra
import logging
from omegaconf import DictConfig
import importlib
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from base.partitioner import DirichletSkewedPartitioner, VariablePathologicalPartitioner
from flwr.server import ServerApp
from base.utils import get_server_fn, get_client_fn
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions

#partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by="label",
#                                   alpha=0.1, min_partition_size=10)
log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
    log_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Saving logs to {log_save_path}")
    client_save_path = (
            f"{log_save_path}/client_states"
        )
    server_save_path = (
            f"{log_save_path}/server_state"
    )
    os.makedirs(client_save_path)
    os.makedirs(server_save_path)
    
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
    model_name = ''.join(word.capitalize() for word in cfg.model.model_class_name.split('_')) # If a model file is mobile_net, the model name is MobileNet
    model_module = getattr(importlib.import_module(f'nets.{cfg.model.model_class_name}'), model_name)
    try:
        model_manager = getattr(importlib.import_module(f'{cfg.algorithm}.model'), f'ModelManager{cfg.algorithm.capitalize()}')  
    except ModuleNotFoundError:
        model_manager = getattr(importlib.import_module(f'base.model'), 'ModelManager')  
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
    
    print(history)

if __name__ == "__main__":
    main()