import subprocess
import os
import argparse
from omegaconf import OmegaConf

sub_path = '/.hydra/config.yaml'

def run_results(log_path):
    cfg = OmegaConf.load(f'{log_path}/{sub_path}')
    for client_id in range(cfg.num_clients):
        cmd = ["python3", "evaluate_model.py", "-l", log_path, '-n', str(client_id)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for {log_path}/test_metrics/test_metrics_{client_id}.json: {e}")

                
def main() -> None:
    parser = argparse.ArgumentParser(description="Get results from model")
    
    parser.add_argument("-l", "--log_directory", type=str, help="Directory to save logs")

    log_directory  = parser.parse_args().log_directory
    
    # we check if metric_0 exists to know if we have to run evaluation or not
    if os.path.exists(f'{log_directory}/{sub_path}') and not os.path.exists(f'{log_directory}/test_metrics/test_metrics_0.json'):
        run_results(f'{log_directory}')
        return
    for dir in os.listdir(log_directory):
        log_path = f'{log_directory}/{dir}'
        if os.path.exists(f'{log_path}/{sub_path}'):
            if not os.path.exists(f'{log_path}/test_metrics/test_metrics_0.json'):
                run_results(log_path)
        else:
            for log_dir in os.listdir(f'{log_directory}/{dir}'):
                log_path = f'{log_directory}/{dir}/{log_dir}'
                if not os.path.exists(f'{log_path}/test_metrics/test_metrics_0.json'):
                    run_results(log_path)
                
if __name__ == '__main__':
    main()