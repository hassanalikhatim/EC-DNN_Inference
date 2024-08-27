import numpy as np
import tenseal as ts



class Dense_Layer:
    
    def __init__(self):
        return
    
    
    def call(self, _tensors, layer, **kwargs):
        
        weights, biases = layer.trainable_variables
        
        return [_tensor.mm(weights.numpy()) + biases.numpy() for _tensor in _tensors]