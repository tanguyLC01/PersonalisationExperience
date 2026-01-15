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
    

class PartialLayerFedAvg(FedAvg):
    """FedAvg strategy that saves the global model after each round."""
    
    def __init__(self, save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_path = save_path
        os.makedirs(self.global_model_path, exist_ok=True)
        self.latest_aggregated = None

    def _save_global_model(self, server_round: int, parameters: Parameters):
        ndarrays = parameters_to_ndarrays(parameters)
        data = {'global_parameters': ndarrays}
        filename = f'{self.global_model_path}/parameters_round_{server_round}.pkl'
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)
    
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
        client_sizes = {}
        

        for client, fit_res in results:
            client_sizes[client.cid] = fit_res.num_examples
            # The layers are ordered the same way in each client and we cut the model in half. At the moment, there can't be a personalised layer between two global layers.
            parameters = parameters_to_ndarrays(fit_res.parameters)
            for i, weights in enumerate(parameters):
                if not i in layer_dict:
                    layer_dict[i] = []
                layer_dict[i].append((np.array(weights), fit_res.num_examples))


        # Aggregate each layer independently
        for i, weighted_layers in layer_dict.items():
            total = sum(size for _, size in weighted_layers)
            avg = np.sum([weight * size for weight, size in weighted_layers], axis=0) / total
            layer_dict[i] = avg
            
        self.latest_aggregated = {i: np.array(layer) for i, layer in layer_dict.items()}
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        # No need to return Parameters as they are different per client. 
        # In Flower, the returned parameter from aggregate fit will be seen as an argument in configure_fit and configure_evaluate
        # return Parameters(tensors=[], tensor_type=""), metrics_aggregated
        log(INFO, f'Number of layers updated : {len(self.latest_aggregated)}')
        params = ndarrays_to_parameters(list(self.latest_aggregated.values()))
        return params, metrics_aggregated
    
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, dict[str, Scalar]] | None:
        """Evaluate the global model and save it."""   
        # Save the global model before evaluation
        self._save_global_model(server_round, parameters)
        
        # Call the original evaluate method        
        return super().evaluate(server_round, parameters)
