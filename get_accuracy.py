import re
import argparse
from omegaconf import OmegaConf
import os
import json
# Chemin vers votre fichier


# Fonction pour extraire les valeurs d'exactitude
def extract_accuracies(log_path):
    accuracies = []
    pattern = re.compile(r'\(\d+,\s*([0-9.]+)\)')
    log_file = OmegaConf.load(f'{log_path}/.hydra/config.yaml')
    algo = log_file.algorithm 
    if algo == 'fedavgft':
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
    print(f"Moyenne des dix dernières valeurs : {mean_accuracy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Get results from model")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")
    
    log_path  = parser.parse_args().log_directory
 
    # Extraction des dix dernières valeurs ou de la valeur moyenne des accuracies apèrs fine-tuning
    extract_accuracies(log_path)

if __name__ == "__main__":
    main()