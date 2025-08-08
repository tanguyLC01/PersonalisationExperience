import re
import ast
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.graph_objs as go
import argparse
import os
from typing import Dict
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go



def extract_metrics(log_path: str) -> Dict[str, Dict]:
    # Initialisation
    metrics = {"fit": defaultdict(list), "evaluate": defaultdict(list)}
    pattern = re.compile(r"History \(metrics, distributed, (fit|evaluate)\):")
    current_section = None

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            current_section = match.group(1)
            # Les métriques sont sur les lignes suivantes, sous forme de dict
            metrics_lines = []
            j = i + 1
            # On récupère toutes les lignes du dict
            while j < len(lines) and not lines[j].strip().endswith("}"):
                metrics_lines.append(lines[j].split(" - ")[-1])
                j += 1
            if j < len(lines):
                metrics_lines.append(lines[j].split(" - ")[-1])
            # On assemble et on parse
            metrics_str = "".join(metrics_lines)
            try:
                metrics_dict = ast.literal_eval(metrics_str)
                for metric_name, values in metrics_dict.items():
                    for round_idx, val in values:
                        metrics[current_section][metric_name].append((round_idx, val))
            except Exception as e:
                print(f"Erreur lors du parsing des métriques : {e}")
    return metrics

def plot_metrics(metrics, save_path: str) -> None:

    # Trouver toutes les métriques présentes
    all_metric_names = set(metrics["fit"].keys()) | set(metrics["evaluate"].keys())

    for metric in all_metric_names:
        fig = go.Figure()

        # Ajout des courbes pour train et test
        for section, label, color in [
            ("fit", "Train", "blue"),
            ("evaluate", "Test", "red")
        ]:
            if metric in metrics[section]:
                rounds, values = zip(*metrics[section][metric])
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=values,
                    mode='lines+markers',
                    name=f"{label} - {metric}",
                    marker=dict(color=color)
                ))

        # Mise en forme
        fig.update_layout(
            title=f"Flower Metrics: {metric.capitalize()} (Train vs Test)",
            xaxis_title="Round",
            yaxis_title=metric.capitalize(),
            legend=dict(x=1.05, y=1),
            width=800,
            height=500,
            margin=dict(l=80, r=80, t=80, b=40)
        )

        # Sauvegarder et afficher
        fig.write_image(f"{save_path}_{metric}.png")
        fig.show(renderer="browser")

    
def main():
    parser = argparse.ArgumentParser(description="Plot Flower Metrics from Log File")
    parser.add_argument("-l", "--log_path", type=str, required=True, help="Path to the log file")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"Le fichier de log {args.log_path} n'existe pas.")
        return

    metrics = extract_metrics(args.log_path)
    save_path = os.path.join(os.path.split(args.log_path)[0], "metric_")
    plot_metrics(metrics, save_path)

if __name__ == "__main__":
    main()

