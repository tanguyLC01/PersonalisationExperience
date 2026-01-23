import importlib
import numpy as np

def load_client_element(cfg):
    model_name = ''.join(word.capitalize() for word in cfg.model.model_class_name.split('_')) # If a model file is mobile_net, the model name is MobileNet
    model_module_class = getattr(importlib.import_module(f'nets.{cfg.model.model_class_name}'), model_name)
    try:
        model_manager_class = getattr(importlib.import_module(f'{cfg.algorithm}.model'), f'ModelManager{cfg.algorithm.capitalize()}')  
    except ModuleNotFoundError:
        model_manager_class = getattr(importlib.import_module(f'base.model'), 'ModelManager')
    try:
        client_class_name = getattr(importlib.import_module(f'{cfg.algorithm}.client'), f'{cfg.algorithm.capitalize()}Client')  
    except ModuleNotFoundError:
        client_class_name = getattr(importlib.import_module(f'base.client'), 'PersonalizedClient') 
    
    return client_class_name, model_manager_class, model_module_class


def load_server_element(cfg):
    try:
        server_class_name = getattr(importlib.import_module(f'{cfg.algorithm}.server'), f'{cfg.algorithm.capitalize()}Server')  
    except ModuleNotFoundError:
        server_class_name = getattr(importlib.import_module(f'base.server'), 'PartialLayerFedAvg') 
    return server_class_name
    