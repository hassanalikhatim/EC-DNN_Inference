from tracemalloc import start
import tenseal as ts
import numpy as np
import time


from server_utils.debugging_server import Debugging_Server

from client_utils.client import Client


class Debugging_Client(Client):
    def __init__(
        self, 
        server :Debugging_Server,
        data_type = 'cv',
        scale=20, depth=5, degree=13,
        pre_compute_layers=[]
    ):
        
        self.server = server
        
        super().__init__(
            server, data_type=data_type,
            scale=scale, depth=depth, degree=degree,
            pre_compute_layers=pre_compute_layers
        )
        
        return
    
    
    def inference(self, x_test):
        
        self.prepare_context(self.scale)
        
        x_test_ = x_test.copy()
        for layer in self.pre_compute_layers:
                x_test_ = layer(x_test_).numpy()
        
        times = []
        all_outputs = []
        all_decrypted_outputs = []
        for n, x_sample in enumerate(x_test_):
            
            pass_str = '\rSample: {:3d}/{:3d} | '.format(n, len(x_test_))
            
            img_height, img_width, channels = x_sample.shape
            _x_sample = x_sample.reshape(img_height*img_width, channels).transpose()
            
            _x_inter = self.encode_tensor(_x_sample)
            
            try:
                start_time = time.time()
                _outputs = self.server.inference(_x_inter, self.get_server_request, pass_str=pass_str)
                end_time = time.time() - start_time
                
                decrypted_outputs = [self.decode_tensor(_output) for _output in _outputs]
                outputs = self.server.unencrypted_inference(x_sample)
                
                assert len(decrypted_outputs) == len(outputs), str(len(decrypted_outputs)) + '=/=' + str(len(outputs))
                
                all_outputs.append(outputs)
                all_decrypted_outputs.append(decrypted_outputs)
                times.append(end_time)
                
            except Exception as e:
                print(pass_str + 'skipping becuase {}'.format(e))
        
        return all_outputs, all_decrypted_outputs, np.array(times)
