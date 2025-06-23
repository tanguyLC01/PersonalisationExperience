import os
import re
import argparse
from omegaconf import OmegaConf
import json
import itertools
import matplotlib.pyplot as plt

regex = re.compile('test_metrics_[0-9]+(.)json')

parser = argparse.ArgumentParser(description="Evaluate Personalized Models")
parser.add_argument("-r", "--root_directory", type=str, default='./', help="Directory to search for files")
rootdir = parser.parse_args().root_directory

results = {}
results['client0'] = {'localier': {'accuracy': [], 'f1-score': []}, 'globalier': {'accuracy': [], 'f1-score': []}, 'baseline': {'accuracy': [], 'f1-score': []}}
results['client1'] = {'localier': {'accuracy': [], 'f1-score': []}, 'globalier': {'accuracy': [], 'f1-score': []}, 'baseline': {'accuracy': [], 'f1-score': []}}
alphas = [0.1, 0.3, 0.6]

for root, dirs, files in os.walk(rootdir):
  for file in files:
    if regex.match(file):
        cfg_dir = dirs[0]
        cfg = OmegaConf.load(f'{root}/{cfg_dir}/config.yaml')
        alpha = cfg.dataset.partitioner.alpha if cfg.dataset.partitioner.name == 'dirichlet' else None
        if alpha > 0.6:
            continue
        model_type = cfg.model.model_type
        num_client = file.split('_')[2].split('.')[0] if 'test_metrics' in file else None
        with open(os.path.join(root, file), 'r') as f:
            data = json.load(f)
        if cfg.training_type == 'personalized':
            results[f'client{num_client}'][model_type]['accuracy'].append(data['accuracy'])
            results[f'client{num_client}'][model_type]['f1-score'].append(data['classification_report']['weighted avg']['f1-score'])
        else:
            results[f'client{num_client}']['baseline']['accuracy'].append(data['accuracy'])
            results[f'client{num_client}']['baseline']['f1-score'].append(data['classification_report']['weighted avg']['f1-score'])

# Plotting the results 
# Plot Accuracy

# Define colors for clients and linestyles for model types
client_colors = {
    'client0': 'tab:blue',
    'client1': 'tab:orange',
}
model_linestyles = {
    'localier': '-',
    'globalier': '--',
    'baseline': ':',
}

# Plot Accuracy
plt.figure(figsize=(12, 6))
for client, metrics in results.items():
    color = client_colors.get(client, None)
    for model_type, values in metrics.items():
        linestyle = model_linestyles.get(model_type, '-')
        plt.plot(
            alphas, 
            values['accuracy'], 
            label=f'{client} - {model_type}', 
            marker='o', 
            color=color, 
            linestyle=linestyle
        )
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Experiment Index')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(rootdir, 'results_accuracy_plot.png'), dpi=300)
plt.close()

# Plot F1-Score
plt.figure(figsize=(12, 6))
for client, metrics in results.items():
    color = client_colors.get(client, None)
    for model_type, values in metrics.items():
        linestyle = model_linestyles.get(model_type, '--')
        plt.plot(
            alphas, 
            values['f1-score'], 
            label=f'{client} - {model_type}', 
            marker='x', 
            color=color, 
            linestyle=linestyle
        )
plt.title('Model F1-Score Comparison')
plt.ylabel('F1-Score')
plt.xlabel('Experiment Index')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(rootdir, 'results_f1score_plot.png'), dpi=300)
plt.close()