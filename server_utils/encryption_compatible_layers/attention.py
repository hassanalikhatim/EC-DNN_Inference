import numpy as np
import tenseal as ts
import tensorflow as tf


from server_utils.encryption_compatible_layers.custom_activation import Custom_Activation_Layer



eps = 0 # 1e-3

class Attention_Layer:
    
    def __init__(self):
        
        self.custom_activation = Custom_Activation_Layer(activation_name='sigmoid_approx')
        
        return
    
    
    def prepare_weights(self, layer):
        
        n, d, channels = layer.input_shape[1:] # img_height x img_width x channels
        
        wq, wk, wv = layer.get_variables()
        wq, wk, wv = wq.numpy(), wk.numpy(), wv.numpy()
        
        q, k, v = wq.shape[1], wk.shape[1], wv.shape[1]
        
        wqk_t = np.dot(wq, np.transpose(wk))
        
        wqk_t_ = np.zeros( (n*d, n*d) )
        wv_ = np.zeros( (n*d, n*v) )
        for i in range(n):
            wqk_t_[ i*d:(i+1)*d, i*d:(i+1)*d ] = wqk_t.copy()
            wv_[ i*d:(i+1)*d, i*v:(i+1)*v ] = wv.copy()
        
        return wqk_t_, wv_
    
    
    def call(self, _tensor, layer, **kwargs):
        
        n, d, channels = layer.input_shape[1:] # img_height x img_width x channels
        wq, _, wv = layer.get_variables()
        q, v = wq.numpy().shape[1], wv.numpy().shape[1]
        
        wqk_t_, wv_ = self.prepare_weights(layer)
        
        _queries_keys = [_tensor_.mm(wqk_t_) for _tensor_ in _tensor]
        
        # _queries_keys = _queries_keys.mm( self.transpose(_tensor_to_process, (n, d)) )
        _queries_keys = kwargs['server_requests'](
            'encrypted_dot_encrypted',
            _tensor_1=_queries_keys, shape_1=(n,d),
            _tensor_2=self.transpose(_tensor, (n,d)), shape_2=(d,n)
        )
        
        _attention_map = self.custom_activation.call(_queries_keys, layer)  # nn length vector
        
        try:
            _values = [_tensor_.mm(wv_) for _tensor_ in _tensor]
        except:
            _values = kwargs['server_requests'](
                'encrypted_dot_unencrypted',
                _tensor_1=_tensor, shape_1=(n,d),
                tensor_2=wv.numpy()
            )
        
        answer = kwargs['server_requests'](
            'encrypted_dot_encrypted',
            _tensor_1=_attention_map, shape_1=(n, n),
            _tensor_2=_values, shape_2=(n, v)
        )
        
        return answer
    
    
    def transpose(self, _flattened_array, flattened_array_shape: tuple):
        
        n, q = flattened_array_shape
        
        transposed_indices = []
        for i in range(q):
            transposed_indices += list(np.arange(i, n*q, q))
        
        transformer = np.zeros( (n*q, n*q) )
        for i in range(n*q):
            transformer[ transposed_indices[i], i ] = 1.
        
        return [_flattened_array_.mm(transformer) for _flattened_array_ in _flattened_array]
    
    
    def attention_layer(self, _tensor, layer, **kwargs):
        
        queries, keys, values = layer.get_variables()
        queries, keys, values = queries.numpy(), keys.numpy(), values.numpy()
        units = keys.shape[1]
        
        queries_keys = np.dot(queries, np.transpose(keys))
        _QK = _tensor.dot(queries_keys)
        _output = _tensor.dot(_QK.dot(_tensor.tranpose()))
        
        activation = self.activation
        self.activation = 'sigmoid'
        _output = self.custom_activation(_output, layer)
        self.activation = activation
        
        _V = _tensor.dot(values)
        
        return _output.dot(_V)
    
        
    