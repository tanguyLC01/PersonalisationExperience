
from base.server import PartialLayerFedAvg
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict
from flwr.server.strategy import FedAvg
import os
import flwr
import pickle    
from collections import defaultdict
import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from logging import WARNING, INFO
from typing import Union, Optional
from flwr.server.client_proxy import ClientProxy


class FedasServer(PartialLayerFedAvg):
    
    def __init__(self, save_path, *args, **kwargs):
        super().__init__(save_path, *args, **kwargs)
        
        
    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
            log(INFO, "Starting Aggregation fit") 
            if not results:
                log(WARNING, "NO RESULTS")
                return None, {}
            
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                log(WARNING, "FAILURES")
                return None, {}
            
            layer_dict = defaultdict(list)

            # We normalize the distribution of the aggregation weights : "alpha"
            alphas = np.array([res.metrics['alpha'] for _, res in results])
            alphas /= alphas.sum()

            for client_number, (_, fit_res) in enumerate(results):
                # The layers are ordered the same way in each client and we cut the model in half. At the moment, there can't be a personalised layer between two global layers.
                parameters = parameters_to_ndarrays(fit_res.parameters)
                for i, weights in enumerate(parameters):
                    if not i in layer_dict:
                        layer_dict[i] = []
                    layer_dict[i].append((np.array(weights), alphas[client_number]))
                    
                     
            # We get the state_dict of the previous server roudn model for the update phase 
            old_state_dict = {}
            filename = f'{self.global_model_path}/parameters_round_{server_round-1}.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                    
                ndarrays = data['global_parameters']
                
                for key, array in zip(layer_dict.keys(), ndarrays):
                    old_state_dict[key] = array


            # Aggregate each layer independently
            for i, weighted_layers in layer_dict.items():
                avg = (1-1/len(parameters)) * old_state_dict[i] +  1/len(parameters) * np.sum([weight * size for weight, size in weighted_layers], axis=0)
                layer_dict[i] = avg
                
            self.latest_aggregated = {i: np.array(layer) for i, layer in layer_dict.items()}
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, {k: res.metrics[k] for k in res.metrics.keys() if k != 'alpha'}) for _, res in results]
                # We remove alpha from the metrics because we don't to aggragate it
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
                
            # No need to return Parameters as they are different per client. 
            # In Flower, the returned parameter from aggregate fit will be seen as an argument in configure_fit and configure_evaluate
            # return Parameters(tensors=[], tensor_type=""), metrics_aggregated
            log(INFO, f'Number of layers updated : {len(self.latest_aggregated)}')
            params = ndarrays_to_parameters(list(self.latest_aggregated.values()))
            return params, metrics_aggregated