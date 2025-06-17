from flwr.client import ClientApp

from flwr.simulation import run_simulation
import numpy as np
from flwr_datasets import FederatedDataset
import os
import hydra
from omegaconf import DictConfig

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.server import ServerApp
from fedper.utils import get_server_fn, get_client_fn
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions
import time

#partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by="label",
#                                   alpha=0.1, min_partition_size=10)

@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    
    root_path = os.getcwd()
    session_name = f"{root_path}/{cfg.session_name}"
    time_identifier = time.strftime("%H-%M-%S")
    log_save_path = f'{session_name}/{time_identifier}'
    client_save_path = (
            f"{log_save_path}/client_states/"
        )
    server_save_path = (
            f"{log_save_path}/server_state"
    )
    os.makedirs(client_save_path)
    os.makedirs(server_save_path)
    
    partitioner = DirichletPartitioner(alpha=0.1, num_partitions=cfg.num_clients, partition_by="label")
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": partitioner})

    _ = plot_label_distributions(partitioner=fds.partitioners["train"],
    label_name="label",
    legend=True,
    )

    plt.savefig(f'{log_save_path}/samples_per_label_per_client.png')
    plt.close()

    
    
    client_fn = get_client_fn(cfg, client_save_path, fds)
    client = ClientApp(client_fn)
    # Create a new server
    # ver instance with the updated FedAvg strategy
    server_fn = get_server_fn(cfg, server_save_path)
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": 
            {"num_cpus": cfg.client_resources.num_cpus, 
             "num_gpus": cfg.client_resources.num_gpus}
            }
    )

if __name__ == "__main__":
    main()