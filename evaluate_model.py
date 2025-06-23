from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import numpy as np
import pickle
from fedper.model import PersonalizedNet
from torchvision import datasets, transforms
from GradCam.utils import generate_grad_cam
from GradCam.hook import LayerHook
import argparse
from omegaconf import OmegaConf
from fedper.utils import load_datasets
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
import json

def load_global_model(net, log_directory, num_rounds):
    with open(f"{log_directory}/server_state/parameters_round_{num_rounds}.pkl", "rb") as f:
        data = pickle.load(f)
            
        ndarrays = data['global_parameters']

        state_dict = net.global_net.state_dict()

        new_state_dict = {}
        for key, array in zip(state_dict.keys(), ndarrays):
            new_state_dict[key] = torch.tensor(array, dtype=torch.float32)
        
        net.global_net.load_state_dict(new_state_dict)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Personalized Models")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    parser.add_argument('-n', '--num_client', type=int, default=0, help='Client Id to evalute')
    parser.parse_args()
    log_directory  = parser.parse_args().log_directory
    client_id = parser.parse_args().num_client
    config_file = f"{log_directory}/.hydra/config.yaml"
    cfg = OmegaConf.load(config_file)
    
    net = PersonalizedNet(cfg.model.num_classes, cfg.model.model_type)
    net.eval()
 
    if cfg.training_type == 'base':
        load_global_model(net, log_directory, cfg.num_rounds)
        
    elif cfg.training_type == 'personalized':
        net.local_net.load_state_dict(torch.load(f"{log_directory}/client_states/local_net_{client_id}.pth"))
        load_global_model(net, log_directory, cfg.num_rounds)  

    # regular_transform= transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(0.5, 0.5)
    # ])

    if cfg.dataset.name == 'fashion_mnist':
        #mnist_test = datasets.FashionMNIST(root='../data', train=False, download=True, transform=regular_transform)
        partitioner = DirichletPartitioner(alpha=0.1, num_partitions=cfg.num_clients, partition_by="label", seed=cfg.seed)
        fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": partitioner})
        # Get the numerical repartition of each class for the selected client
        full_dataset = fds.load_split("train")
        fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True)
        # Retrieve the class names
        class_names = fashion_mnist_test.classes
        
        labels_train = np.array(full_dataset["label"])
        _, count_train = np.unique(labels_train, return_counts=True)
        
        train_dataset = fds.load_partition(client_id, "train")
        labels = np.array(train_dataset["label"])
        unique, counts = np.unique(labels, return_counts=True)
        print("Class repartition for client", client_id)
        for cls, count in zip(unique, counts):
            print(f"Class {cls} ({class_names[int(cls)]}): {count/count_train[int(cls)]:.2f} samples")
        # Everyone see the same test set / Loads_slipt in load_datasets enable to load the test set entirely
        test_loader = load_datasets(client_id, fds, cfg)[2]
        fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True)

        
        # Retrieve the class names
        class_names = fashion_mnist_test.classes
        
    #test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=cfg.client_config.batch_size, shuffle=True)


    net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch['image'], batch['label']
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    # Save metrics to log file
    metrics = {
        "classification_report": report,
        "accuracy": report['accuracy']
    }
    metrics_path = f"{log_directory}/test_metrics_{client_id}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Per-class metrics:")
    for cls, metrics in report.items():
        if cls.isdigit():
            print(f"Class {cls} ({class_names[int(cls)]}): Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, Support={metrics['support']}")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    
if __name__ == "__main__":
    main()

#def difference_in_heatmap(net_1, net_2, loader):
#     net_1.eval()
#     net_2.eval()
#     res_global = []
#     res_local = []
    
#     extractor_conv1_1 = LayerHook(net_1, 'conv1', 'global_net')
#     extractor_conv1_2 = LayerHook(net_2, 'conv1', 'global_net')
#     extractor_conv2_1 = LayerHook(net_1, 'conv2', 'local_net')
#     extractor_conv2_2 = LayerHook(net_2, 'conv2', 'local_net')
    
#     for data, _ in loader:
#         class_1 = net_1(data).topk(1).indices.cpu().numpy()[0]
#         class_2 = net_2(data).topk(1).indices.cpu().numpy()[0]
#         heatrmpa_conv1_1 = generate_grad_cam(net_1, data, extractor_conv1_1, class_1)
#         heatmap_conv1_2 = generate_grad_cam(net_2, data, extractor_conv1_2, class_2)
#         heatmap_conv2_1  = generate_grad_cam(net_1, data, extractor_conv2_1, class_1)
#         heatmap_conv2_2  = generate_grad_cam(net_2, data, extractor_conv2_2, class_2)
        
#         temp_global = np.abs(heatrmpa_conv1_1 - heatmap_conv1_2)
#         temp_global = np.nan_to_num(temp_global, nan=0.0)  # Handle NaN values
#         res_global.append(np.mean(temp_global))
        
#         temp_local = np.abs(heatmap_conv2_1 - heatmap_conv2_2)
#         temp_local = np.nan_to_num(temp_local, nan=0.0)
#         res_local.append(np.mean(temp_local))
        
    
#     return sum(res_global) / len(res_global), sum(res_local) / len(res_local)