import importlib
import numpy as np

def load_client_element(cfg):
    model_name = ''.join(word.capitalize() for word in cfg.model.model_class_name.split('_')) # If a model file is mobile_net, the model name is MobileNet
    model_module_class = getattr(importlib.import_module(f'nets.{cfg.model.model_class_name}'), model_name)
    try:
        model_manager_class = getattr(importlib.import_module(f'{cfg.algorithm}.model'), f'ModelManager{cfg.algorithm.capitalize()}')  
    except ModuleNotFoundError:
        model_manager_class = getattr(importlib.import_module(f'base.model'), 'ModelManager')  
    client_class_name = getattr(importlib.import_module(f'base.client'), cfg.client_config.client_class_name)
    
    return client_class_name, model_manager_class, model_module_class