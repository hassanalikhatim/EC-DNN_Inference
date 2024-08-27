import numpy as np
import tenseal as ts



class Server_Functions:
    
    def __init__(self):
        return
    
    
    def get_server_request(self, request, **kwargs):
        
        if request == 're_encrypt':
            return self.re_encrypt(kwargs['_tensor'])
        elif request == 'encrypted_dot_encrypted':
            return self.encrypted_dot_encrypted(
                kwargs['_tensor_1'], kwargs['shape_1'],
                kwargs['_tensor_2'], kwargs['shape_2']
            )
        elif request == 'encrypted_dot_unencrypted':
            return self.encrypted_dot_unencrypted(
                kwargs['_tensor_1'], kwargs['shape_1'],
                kwargs['tensor_2']
            )
        elif request == 'transpose_and_flatten':
            return self.transpose_and_flatten(kwargs['_tensor'])
        else:
            assert False, 'Ivalid request.'
            
        return
    
    
    def transpose_and_flatten(self, _tensor):
        
        decoded = self.decode_tensor(_tensor).transpose().reshape(1, -1)
        
        return self.encode_tensor(decoded)
    
    
    def re_encrypt(self, _tensor):
        
        decoded = self.decode_tensor(_tensor)
        
        if len(decoded.shape) == 1:
            decoded = decoded.reshape(1, -1)
        
        return self.encode_tensor(decoded)
    
    
    def reshape(self, _tensor, shape: tuple):
        
        decoded = self.decode_tensor(_tensor)
        
        decoded = np.reshape(decoded, shape)
        
        return self.encode_tensor(decoded)
    
    
    def extend_tensor(self, _tensor, source_shape: tuple, target_shape: tuple):
        
        n, d = source_shape
        nn, nd = target_shape
        
        tensor = self.decode_tensor(_tensor)
        
        extended_tensor = np.zeros( (nn, nd) )
        for i in range(n):
            extended_tensor[ i*n:(i+1)*n, i*d:(i+1)*d ] = tensor.copy()
            
        return extended_tensor
    
    
    def encrypted_dot_encrypted(self, _tensor_1, shape_1: tuple, _tensor_2, shape_2: tuple):
        
        tensor_1 = self.decode_tensor(_tensor_1)
        tensor_2 = self.decode_tensor(_tensor_2)
        
        answer = np.dot(tensor_1.reshape(shape_1), tensor_2.reshape(shape_2))
        
        return self.encode_tensor(answer.reshape(1, -1))
    
    
    def encrypted_dot_unencrypted(self, _tensor_1, shape_1: tuple, tensor_2):
        
        tensor_1 = self.decode_tensor(_tensor_1)
        
        answer = np.dot(tensor_1.reshape(shape_1), tensor_2)
        
        return self.encode_tensor(answer.reshape(1, -1))