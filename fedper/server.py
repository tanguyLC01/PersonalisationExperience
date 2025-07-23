from typing import List, Tuple, Dict
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import os
import flwr
import pickle    
from collections import defaultdict
import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from logging import WARNING, INFO
from typing import Union, Optional
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from sympy import I


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    

class PartialLayerFedAvg(FedAvg):
    """FedAvg strategy that saves the global model after each round."""
    
    def __init__(self, save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_path = save_path
        os.makedirs(self.global_model_path, exist_ok=True)
        self.latest_aggregated = None

    def _save_global_model(self, server_round: int, parameters):
        ndarrays = flwr.common.parameters_to_ndarrays(parameters)
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
            avg = sum(weight * size for weight, size in weighted_layers) / total
            layer_dict[i] = avg
            
        self.latest_aggregated = layer_dict
        # Collect all global_net weights and number of examples
        # weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        # num_examples = [fit_res.num_examples for _, fit_res in results]

        # aggregated = []
        # for params in zip(*weights):
        #     weighted_sum = sum(w * n for w, n in zip(params, num_examples))
        #     total = sum(num_examples)
        #     aggregated.append(weighted_sum / total)

        #self.latest_aggregated = aggregated
         # Aggregate custom metrics if aggregation fn was provided
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
        params = ndarrays_to_parameters(self.latest_aggregated.values())
        return params, metrics_aggregated
    
    # def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> list[tuple[ClientProxy, FitIns]]:
    #     config = {}
    #     if self.on_fit_config_fn is not None:
    #         # Custom fit config function provided
    #         config = self.on_fit_config_fn(server_round)
            
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
        
        
        # instructions = []
        # for client  in client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients):
            
        #     if self.latest_aggregated is not None: # We already have done a round, we know which layers each client sents
        #         layers_to_send = self.layers_per_client[client.cid]
        #         tensors = [self.latest_aggregated[name] for name in layers_to_send]
        #         params = Parameters(tensors=[t.tolist() for t in tensors], tensor_type="numpy")
        #         config['layers'] =  layers_to_send
        #     else:
        #         params = parameters
        #     instructions.append((client, FitIns(parameters=params, config=config)))

        # return instructions
    
    def evaluate(self, server_round: int, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the global model and save it."""   
        # Save the global model before evaluation
        self._save_global_model(server_round, parameters)
        
        # Call the original evaluate method        
        return super().evaluate(server_round, parameters)
