from tracemalloc import start
import tenseal as ts
import numpy as np
import time

from utils_.general_utils import get_memory_usage



class Tester:
    def __init__(self, client, server):
        self.client = client
        self.server = server
        
        
    def test_efficient(self, x_test, y_test, batch_size=32):
        total_batches = int(len(x_test)/batch_size)
        
        self.client.prepare_context()
        
        decrypted_outputs = []
        for batch in range(total_batches):
            x_inter = np.copy(x_test[batch*batch_size:(batch+1)*batch_size]).reshape(batch_size, -1)
            _x_inter = self.client.encode_tensor(x_inter)
            
            for l, layer in enumerate(self.server.model.layers[:-1]):
                for layer_name in self.server.layer_names:
                    if layer_name in layer.name:
                        # Getting % RAM usage
                        print('\r', batch, '/', total_batches, '\t', 
                              l, ',', layer_name, 
                              ': RAM memory % used:', get_memory_usage(), 
                              end='')
                        try:
                            _x_inter = self.server.layer_funcs[layer_name](_x_inter, layer)
                        except:
                            x_inter = self.client.decode_tensor(_x_inter).reshape(batch_size, -1)
                            _x_inter = self.client.encode_tensor(x_inter)
                            _x_inter = self.server.layer_funcs[layer_name](_x_inter, layer)         
            decrypted_outputs.append(self.client.decode_tensor(_x_inter))
            
        return np.array(decrypted_outputs).reshape(len(x_test), -1)
    
    
    def test_one(self, x_test):
        self.client.prepare_context()      
        _x_inter = self.client.encode_tensor(x_test.reshape(1, -1))
        _x_inter = self.active_inference(_x_inter)
        
        return self.client.decode_tensor(_x_inter)[0]
    
    
    def active_inference(self, _x_inter):
        for l, layer in enumerate(self.server.model.layers[:-1]):
            for layer_name in self.server.layer_names:
                if layer_name in layer.name:
                    _x_inter = self.server.layer_funcs[layer_name](self.refresh(_x_inter), layer)
        return _x_inter
                        
    
    def refresh(self, _x_inter):
        x_inter = self.client.decode_tensor(_x_inter).reshape(1, -1)
        _x_inter = self.client.encode_tensor(x_inter)
        return _x_inter
    
    
    
    
    

    #################################################################################################    
    # Not being used in the code. Ignore this function.
    def test(self, x_test, y_test):
        encryption_times = np.zeros((len(x_test)))
        inference_times = np.zeros((len(x_test)))
        actual_outputs = np.zeros((len(x_test), y_test.shape[-1]))
        decrypted_outputs = np.zeros((len(x_test), y_test.shape[-1]))
        layer_wise_differences = np.zeros((len(x_test), len(self.server.model.layers)))
        
        self.client.prepare_context()
        for n, x_sample in enumerate(x_test):
            x_inter = np.copy(x_test[n:n+1])
            _x_inter = self.client.encode_tensor(x_inter[0,:,:,0])
            inference_time = 0
            for l, layer in enumerate(self.server.model.layers[:-1]):
                for layer_name in self.server.layer_names:
                    if layer_name in layer.name:
                        print(l, layer_name, 'RAM memory % used:', get_memory_usage()) # Getting % RAM usage
                        x_inter = layer(x_inter)
                        start_time = time.time()
                        _x_inter = self.server.layer_funcs[layer_name](_x_inter, layer)
                        inference_time += time.time() - start_time
                        #layer_wise_differences[n, l] = np.sum((self.client.decode_tensor(_x_inter) - x_inter)**2)
            inference_times[n] = inference_time
            actual_outputs[n] = x_inter
            decrypted_outputs[n] = self.client.decode_tensor(_x_inter)
        return actual_outputs, decrypted_outputs, layer_wise_differences, encryption_times, inference_times
