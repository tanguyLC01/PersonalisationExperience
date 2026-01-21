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
For example, if we want to launch a FedAvg experiment on Cifar-10 with a Dirichlet partitioner, the conf/config.yaml could look like this :
```
num_clients: 100
num_rounds: 300
seed: 42
session_name: dirichlet_fedavg
algorithm: fedavg
device: cuda

server_config:
  fraction_fit: 0.1
  fraction_evaluate: 1
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2

hydra:
  run:
    dir: ../${session_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}s


defaults:
  - dataset: cifar10
  - client_config: fedavg
  - model: cnn_net
```
This will load 100 clients with a client config specified in `conf/client_config/fedavg.yaml` which all have a model specified in `conf/model/cnn_net.yaml`

If we would like to launch a FedPer experiement, we only have to change the `cnn_net.yaml` config. Indeed, FedPer is the FedAvg algorithm with a the body/backbone of each client's model. Hence, the  `cnn_net.yaml` would look like this :
```
model_class_name: 'cnn_net'
num_classes: 10
personalisation_level: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
in_channels: 3
```
This will specify a personalisation_level - the number of personalized layers starting from the output of the model  - for each client. Indeed, with the `PartialLayerFedAvg`, the server enables clients to send different number of layers depending of their personalization level. In the example below, each client will see its last layer personalized.<br>
NB : You will note here that we specify the in_channels (the number of channels in the input image) in the model setting but this is linked to the dataset choice.
# Reflexivity as Modularity
In order to provide developers with a full managable framework, it is quite easy to implement new strategy and architecture inside this repository.<br>
Let's say you want to implement the <strong>FedRep </strong> method. You need to implement a new `train` method in the ModelManager to be able to separate the training phase between the body and the head.<br>
Hence, we create a sub-directory named fedrep - concordance of the names is crucial - and we create a file `model.py` in this subdirectory. In this last file, we define a class `ModelManagerFedrep` which derives from `base.ModelManager` and we can redefine the the train method in this class.<br>
In the repo, the method `load_classname.load_client_element` handle the reflexivity paradigm. For example, if we specify in the config file `algortihm: fedper`, the function will look for `fedper.model.py`, `fedper.client.py`, .. And, then if it exists, it will choose the method(s) definied in the `base` module.<br>

# Setup
## Environment Installation
```
python3 -m venv ./personalisation_env
source ./personalisation_env/bin/activate
pip install -r requirements.txt
```

# Example
In this section, we present different possible configuration to run experiments.
```bash
python run.py (--config-name path/conf/file)
```
## FedAvg on CIFAR-10
We run the FedAvg algorithm on the CIFAR-10 dataset with a CNN-based model
### base.yaml
```
num_clients: 100
num_rounds: 100
seed: 42
session_name: fedavg_cifar10
algorithm: fedavg
device: cuda

server_config:
  fraction_fit: 0.1
  fraction_evaluate: 1
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2

hydra:
  run:
    dir: ../${session_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}s


defaults:
  - dataset: cifar10
  - client_config: fedavg
  - model: cnn_net
```

### cnn_net.yaml
```
model_class_name: cnn_net
num_classes: 10
personalisation_level: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
in_channels: 3
```
Here, all personalisation levels are at 0 because we want every model to have a full global_net. In this case,locak_net will be a torch Identity Layer.

### client_config/fedavg.yaml
```
client_class_name: BaseClient
num_epochs: 10 
batch_size: 10
learning_rate: 0.01
optimiser: "sgd"
num_cpus: 10
num_gpus: 1
momentum: 0.5
weight_decay: 1e-4 
```
In this config file, we define all the client parameters for training. We choose BaseClient because we can send all the models's parameters to the server as we are doing FedAvg

## FedPer on CIFAR-10
To run FedPer, it is basically the same configuration as the algorithm on the server is the same but it is on the personalisation_lvel for each client, we'll have a role to play. Hence, we change only the `model/{name_of_the_model}.yaml` file to put in place th personalization in fedper.
### cnn_net.yaml
```
model_class_name: cnn_net
num_classes: 10
personalisation_level: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
in_channels: 3
```
Here, we reconstructed the FedPer experiment setting where all client have their final layer personalized.

# Evaluation
Once the federated process is terminated, we can parse the log files to obtain the metrics curves of the experiment
```
python3 get_metrics.py -l path/to/log # With initial config, ${session_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}/run.log
```
Also, it is possible to get the metrics per class and per client with `get_results.py` which will run `evaluate_model.py` for every client of the experiment.
```
python3 get_results.py -l path/to/log_directory
```