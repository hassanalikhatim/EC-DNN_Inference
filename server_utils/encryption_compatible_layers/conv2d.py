import tenseal as ts
import numpy as np



class Conv2D_Layer:
    
    def __init__(self):
        
        self.processing = 0
        self.total_processing = 1
        
        return
    
    
    def prepare_weights(self, layer):
        
        img_height, img_width, _ = layer.input_shape[1:] # img_height x img_width x channels
        hw = img_height*img_width
        
        x_stride, y_stride = layer.strides
        
        weights, _ = layer.trainable_variables
        k0, k1, i_channels, n_filters = weights.numpy().shape # k_size x k_size x i_channel x n_filters
        k0_h, k1_h = int(k0/2), int(k1/2)
        
        self.bigger_kernel_weights = np.zeros( (n_filters, i_channels, hw, hw) )
        for ch in range(i_channels):
            for fn in range(n_filters):
                for k in range( img_height ):
                    for i in range( img_width ):
                        local_kernel = np.zeros((img_height+2*k0_h, img_width+2*k1_h))
                        local_kernel[k:k+k0, i:i+k1] = weights[:, :, ch, fn].numpy().copy()
                        local_kernel = local_kernel[k1_h:img_height+k1_h, k0_h:img_width+k0_h]
                        
                        self.bigger_kernel_weights[ fn, ch, :, k*img_width+i ] = local_kernel.reshape(-1)[:hw]
        
        return
    
    
    # Optimized
    def call(self, _tensor, layer, pass_str='', **kwargs):
        """
        _tensor: channels x img_height*img_width
        layer: keras Conv2D layer
        """
        
        weights, biases = layer.trainable_variables
        _, _, i_channels, n_filters = weights.numpy().shape # k_size x k_size x i_channel x n_filters
        
        self.processing = 0
        self.total_processing = i_channels*n_filters
        
        self.prepare_weights(layer)
        print('\rPrepared weights.', end='')
        
        _processed_tensors = []
        for fn in range(n_filters):
            for ch in range(i_channels):
                
                if ch == 0:
                    _processed_tensor = _tensor[ch].mm(self.bigger_kernel_weights[fn, ch])
                else:
                    _processed_tensor += _tensor[ch].mm(self.bigger_kernel_weights[fn, ch])
                self.update_processing(pass_str)
            
            _processed_tensor += biases[fn] * np.ones( _processed_tensor.shape )
            _processed_tensors.append(_processed_tensor)
            
        return _processed_tensors
    
    
    def update_processing(self, pass_str='\r'):
        
        self.processing += 1
        print(pass_str + ' Processing: {:.2f}% | '.format(100*self.processing/self.total_processing), end='')
        
        return
    
    
    def np_call(self, tensor, layer, **kwargs):
        
        weights, biases = layer.trainable_variables
        _, _, i_channels, n_filters = weights.numpy().shape # k_size x k_size x i_channel x n_filters
        
        self.processing = 0
        self.prepare_weights(layer)
        self.total_processing = i_channels*n_filters
        print('\rPrepared weights.', end='')
        
        processed_tensors = []
        for fn in range(n_filters):
            for ch in range(i_channels):
                
                if ch == 0:
                    processed_tensor = np.dot(tensor[ch], self.bigger_kernel_weights[fn, ch])
                else:
                    processed_tensor += np.dot(tensor[ch], self.bigger_kernel_weights[fn, ch])
                
            processed_tensors.append(processed_tensor + biases[fn])
                
        return processed_tensors
    
    