import tenseal as ts
import numpy as np
import time


from utils_.general_utils import get_memory_usage

from server_utils.encrypted_server import Server



class Debugging_Server(Server):
    
    def __init__(self, model):
        
        super().__init__(
            model
        )
        
        return
    
    
    def prepare_dimensions(self):
        self.height, self.width, self.channels = self.model.input_shape[1:]
        return
    
    
    def inference(self, _x_test, server_requests=None, pass_str='\r', **kwargs):
        
        _x_inter = _x_test
        
        _x_inters = []
        for l, layer in enumerate(self.model.layers[:-1]):
            for layer_name in self.layer_funcs.keys():
                if layer_name in layer.name:
                    pass_str_local = pass_str + 'Layer: {:2d} ({}) | '.format(l, layer_name)
                    pass_str_local += 'RAM memory used: {:.2f}% | '.format(get_memory_usage())
                    print(pass_str_local, end='') # Getting % RAM usage
                    
                    try:
                        _x_inter = self.layer_funcs[layer_name].call(
                            _x_inter, layer, pass_str=pass_str_local, server_requests=server_requests
                        )
                    except:
                        _x_inter = server_requests('re_encrypt', _tensor=_x_inter)
                        _x_inter = self.layer_funcs[layer_name].call(
                            _x_inter, layer, 
                            pass_str=pass_str_local, server_requests=server_requests
                        )
                        
                    _x_inters.append(_x_inter)
            
        return _x_inters
    
    
    def unencrypted_inference(self, x_test):
        
        x_inter = np.expand_dims(x_test, axis=0)
        
        x_inters = []
        for l, layer in enumerate(self.model.layers[:-1]):
            for layer_name in self.layer_funcs.keys():
                if layer_name in layer.name:
                    x_inter = layer(x_inter)
            
                    x_inters.append(x_inter.numpy())
            
        return x_inters