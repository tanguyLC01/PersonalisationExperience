import torch.nn as nn
import torch.nn.functional as F


class LayerHook:
    def __init__(self, model, target_layer, model_type='global_net'):
        self.model = model
        self.target_layer_name = target_layer
        self.feature_map = None
        self.gradient = None
        self.model_type = model_type
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.feature_map = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()
    
        # Register hooks on the specified layer
        layer = dict(dict(self.model.named_children())[self.model_type].named_children())[self.target_layer_name]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)