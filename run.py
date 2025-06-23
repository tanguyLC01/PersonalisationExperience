from flwr.client import ClientApp
from flwr.simulation import run_simulation
import numpy as np
from flwr_datasets import FederatedDataset
import os
import hydra
import logging
from omegaconf import DictConfig

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.server import ServerApp
from fedper.utils import get_server_fn, get_client_fn
import matplotlib.pyplot as plt
from flwr_datasets.visualization import plot_label_distributions

#partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by="label",
#                                   alpha=0.1, min_partition_size=10)
log = logging.getLogger(__name__)
@hydra.main(config_path='conf', config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    
    log_save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Saving logs to {log_save_path}")
    client_save_path = (
            f"{log_save_path}/client_states/"
        )
    server_save_path = (
            f"{log_save_path}/server_state"
    )
    os.makedirs(client_save_path)
    os.makedirs(server_save_path)
    
    if cfg.dataset.partitioner.name == "dirichlet":
        partitioner = DirichletPartitioner(alpha=cfg.dataset.partitioner.alpha, num_partitions=cfg.num_clients, partition_by="label", seed=cfg.seed)
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": partitioner})

    fig, _, _ = plot_label_distributions(partitioner=fds.partitioners["train"],
    label_name="label",
    legend=True,
    )
    
    fig.legend()
    fig.set_size_inches(12, 8)

    fig.savefig(f'{log_save_path}/samples_per_label_per_client.png', dpi=300)
    plt.close()

    
    # Create a new client
    client_fn = get_client_fn(cfg, client_save_path, fds, log)
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