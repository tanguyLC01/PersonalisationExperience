import re
import argparse
from omegaconf import OmegaConf
import os
import json
import subprocess
import numpy as np

# Fonction pour extraire les valeurs d'exactitude
def extract_accuracies(log_path):
    accuracies = []
    pattern = re.compile(r'\(\d+,\s*([0-9.]+)\)')
    log_file = OmegaConf.load(f'{log_path}/.hydra/config.yaml')
    algo = log_file.algorithm 
    if algo == 'fedavgft' or algo == 'fully_local':
        sub_path = f'{log_path}/test_metrics'
        for file in os.listdir(sub_path):
            with open(f'{sub_path}/{file}', 'r') as file:
                metric_file = json.load(file)               
                accuracies.append(metric_file['accuracy'])
        mean_accuracy = sum(accuracies) / len(accuracies)
    else:
        file_path =f'{log_path}/run.log'
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    accuracies.append(float(match.group(1)))
        last_ten = accuracies[-10:] if len(accuracies) >= 10 else accuracies
        mean_accuracy = sum(last_ten) / len(last_ten) if last_ten else 0
    return mean_accuracy


def get_metrics(log_path: str) -> None:
    
    metric_path = os.path.join(log_path, 'test_metrics')
    if not os.path.exists(metric_path):
        cmd = ["python3", "get_results.py", "-l", log_path]
        try:
            subprocess.run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for {log_path}: {e}")
        
    # Extraction des dix dernières valeurs ou de la valeur moyenne des accuracies apèrs fine-tuning
    mean_accuracy = extract_accuracies(log_path)*100
    client_accuracies = []
    for metric_dir in os.listdir(metric_path):
        with open(os.path.join(metric_path, metric_dir), 'r') as f:
            data = json.load(f)
        
        client_accuracies.append(data['accuracy']*100)
    std_accuracies = np.std(np.array(client_accuracies))
    return mean_accuracy, std_accuracies
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get results from model")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    
    log_path  = parser.parse_args().log_directory
 
    mean_accuracy, std_accuracies = get_metrics(log_path)
    print(f'Mean Accuracy : {mean_accuracy} \n Deviation : {std_accuracies}')