import re
import argparse
from omegaconf import OmegaConf
import os
from typing import Tuple
import json
import subprocess
import numpy as np
import sys

# Fonction pour extraire les valeurs d'exactitude
def extract_accuracies(log_path):
    accuracies = []
    log_file = OmegaConf.load(f'{log_path}/.hydra/config.yaml')
    algo = log_file.algorithm 
    if algo == 'fully_local':
        sub_path = f'{log_path}/test_metrics'
        for file in os.listdir(sub_path):
            with open(f'{sub_path}/{file}', 'r') as file:
                metric_file = json.load(file)               
                accuracies.append(metric_file['accuracy'])
        mean_accuracy = sum(accuracies) / len(accuracies)
    elif algo == 'fedavgft':
        pattern = re.compile(f'Epoch {log_file.client_config.finetuned_epoch}/{log_file.client_config.finetuned_epoch}, Loss: [\d\.]+", Accuracy : ([\d\.]+)\)')
        file_path =f'{log_path}/run.log'
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    accuracies.append(float(match.group(1)))
        mean_accuracy = sum(accuracies) / len(accuracies)
    else:
        pattern = re.compile(r'\(\d+,\s*([0-9.]+)\)')
        file_path =f'{log_path}/run.log'
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    accuracies.append(float(match.group(1)))
        last_ten = accuracies[-10:] if len(accuracies) >= 10 else accuracies
        mean_accuracy = sum(last_ten) / len(last_ten) if last_ten else 0
    return mean_accuracy


def get_metrics(log_path: str) -> Tuple[float, float]:
    
    metric_path = os.path.join(log_path, 'test_metrics')
    if not os.path.exists(metric_path):
        print("No test metrics found, running evaluation script...")
        cmd = [sys.executable, "get_results.py", "-l", log_path]
        try:
            subprocess.run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for {log_path}: {e}")
        
    mean_accuracy = extract_accuracies(log_path)*100
    client_accuracies = []
    for metric_dir in os.listdir(metric_path):
        with open(os.path.join(metric_path, metric_dir), 'r') as f:
            data = json.load(f)
        
        client_accuracies.append(data['accuracy']*100)
    std_accuracies = float(np.std(np.array(client_accuracies)))
    return mean_accuracy, std_accuracies
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get results from model")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    
    log_path  = parser.parse_args().log_directory
 
    mean_accuracy, std_accuracies = get_metrics(log_path)
    print(f'Mean Accuracy : {mean_accuracy} \n Deviation : {std_accuracies}')