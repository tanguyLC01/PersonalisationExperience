# Project Overview
The goal of this project is to create a Federated Learning strategy agnostic framework to be able to run smoothly any decomposition based algorithm by only implementing the new strategy.
## Goals
- Run Centralised Federated Learning rounds with personalised models, specific aggregation/training strategy (e.g FedPer, FedAvg, FedAvg+Ft, ...) 
- Automate metrics and plots generation
- Use at maximum the configuration files to specify a new strategy/model/technic.
## Architecture
The `base` module implement the foundation of the Flower strategy. <br>
In `client.py`, we have the _BaseClient_ class that implements the classical logic of FL and _PersonalisedClient_ class enables to have a client that only send its global parameter.<br>
In `partitioner.py`, we have multiple partitioners implemented and a utilitary function to load the desired partitioner from a DictConfig.<br>
In `server.py`, we can find a derived of the FedAvg class that save the global parameters at each round and can compute specific metrics.
### Model class
<div style="display: flex; align-items: center;">
  <img src="model_fig.png" alt="Description" style="width: 150px; margin-right: 20px;">
  <p>The whole benefit of this project lies in the <i>model</i> class.<br>
Every client will not see directly a model but they will own a model manager.<br>
The model manager is the class that implements the strategy of the model (training, testing and personalization handling).<br>
The model split class is only a class wrapped around the model so that any other class sees the Deep Neural Network (DNN) as a real split between local and global nets. </p>
</div>

## Configuration
The configuration files are in the `conf` folder. We use Hydra system to be able to handle multiple configurations at once.<br>
The main usage is to create/change the configurations in `client_config`, `dataset`, `model` to fit to desire strategy and then, go to the `base` file to change the default values to the corresponding modified/new sub-files.<br>
For example, if we want to launch a FedRep experiment on cifar-10 with a dirichlet partitioner :
<ul>
<li> We will change the value of <i>algorithm</i> and <i>client_config</i> to <i>fedrep</i> in base.yaml.</li>
<li> In <i>dataset.cifar10.yaml</i>, we will change the value of the <i>partitioner</i> to <i>dirichlet</i> </li>
</ul>

## Usage
If the strategy we want to use is already predefined, we <strong>only</strong> have to modify the configuration files.<br>
Otherwise, it is necessary to create a module with the name of the algorithm - <strong>this name will be the one in the algorithm field in the `base.yaml` file.</strong><br>
Inside this module, you can derive from the `base` module some of the class - ModelManager, Client, .. - to adapt the code to the new strategy you want to implement.<br>
You can find an example of this process in the `fedrep` and `fedavgft` module.<br>
<strong>WARNING</strong> : Without inheritance from the `base` class, the whole module does not work.


## Troubleshooting

## Contributing

## License

## Contact