from tracemalloc import start
import tenseal as ts
import numpy as np
import time


from client_utils.server_shared_functions import Server_Functions

from server_utils.encrypted_server import Server



class Client(Server_Functions):
    def __init__(
        self, 
        server :Server,
        data_type = 'cv',
        scale=20, depth=5, degree=13,
        pre_compute_layers=[]
    ):
        
        self.server = server
        
        self.data_type = data_type
        self.pre_compute_layers = pre_compute_layers
        
        self.scale = scale
        self.depth = depth
        self.degree = degree
        
        print("Client created.")
        
        return
    
    
    def prepare_context(self, scale):
        
        success = False
        while not success:
            try:
                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=2**self.degree,
                    coeff_mod_bit_sizes=[60]+[self.scale]*(self.depth-2)+[60]
                )
                success = True
                print("success with scale: ", self.scale)
            except:
                self.scale = self.scale + 1
        success = False
        
        self.context.global_scale = 2**self.scale
        self.context.generate_galois_keys()
        
        return
        
    
    def encode_tensor(self, tensor):
        
        info = "Tensor to be encrypted must be 2-dimensional."
        assert len(tensor.shape) == 2, info
        
        return [ts.ckks_vector(self.context, tensor_) for tensor_ in tensor]
    
    
    def decode_tensor(self, _tensor):
        return np.array([_tensor_.decrypt() for _tensor_ in _tensor])
    
    
    def inference(self, x_test):
        self.prepare_context(self.scale)
        
        x_test_ = x_test.copy()
        for layer in self.pre_compute_layers:
                x_test_ = layer(x_test_).numpy()
        
        decrypted_outputs = []
        for n, x_sample in enumerate(x_test_):
            
            img_height, img_width, channels = x_sample.shape
            _x_sample = x_sample.reshape(img_height*img_width, channels).transpose()
            
            _x_inter = self.encode_tensor(_x_sample)
            
            _output = self.server.inference(_x_inter, self.get_server_request)
            
            decrypted_outputs.append(self.decode_tensor(_output))
        
        return decrypted_outputs
