import numpy as np
import tenseal as ts



class Flatten_Layer:
    
    def __init__(self):
        return
    
    
    def call(self, tensor, layer, **kwargs):
        """
        There seems to be a bug in the flatten layer. We are trying to resolve it.
        """
        
        return kwargs['server_requests']('transpose_and_flatten', _tensor=tensor)
        # return [ts.CKKSVector.pack_vectors(tensor)]