import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_classname import load_client_element
import numpy as np
import random
from flwr_datasets import FederatedDataset
import os
import hydra
from omegaconf import DictConfig
from base.utils import load_datasets
from base.partitioner import load_partitioner
import logging
from logging import INFO

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
    
    # --------------------- Choose the right Partitioner (Train and test will have the same) ---------------------
    train_partitioner, test_partitioner = load_partitioner(cfg)
        
    fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": train_partitioner, "test":  test_partitioner})
    
    epochs = int(cfg.client_config.num_epochs * cfg.num_rounds * cfg.server_config.fraction_fit)
    print(epochs)
    for client_id in range(cfg.num_clients):
        log.info(f"------------------ Training Client {client_id} ------------------")
        trainloader, _, testloader  = load_datasets(client_id, fds, cfg)
        log.info(f"Length trainloader : {len(trainloader)} ")
        client_classname, model_manager_class, model_module_class = load_client_element(cfg)
        model_manager  = model_manager_class(client_id=client_id, config=cfg, trainloader=trainloader, testloader=testloader, model_class=model_module_class, client_save_path=f"{client_save_path}/local_net_{client_id}.pth")
        client = client_classname(client_id, model_manager, cfg)
        log.info(f'Epochs : {epochs}')
        client.epochs = epochs
        client.perform_train(verbose=True)

    
        # # Plot Loss and Accuracy
        # plt.figure(figsize=(12, 5))

        # # Loss Plot
        # plt.subplot(1, 2, 1)
        # plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Over Epochs')
        # plt.legend()
        
        # # Accuracy Plot
        # plt.subplot(1, 2, 2)
        # plt.plot(range(1, epochs + 1), val_accuracies, marker='s', linestyle='-', color='r', label='Test Accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.title('Test Accuracy Over Epochs')
        # plt.legend()
        
        # plt.savefig(f'{log_save_path}/train_loss_val_accuracy_{client_id}')
   
    
if __name__ == "__main__":
    main()

        
        

    