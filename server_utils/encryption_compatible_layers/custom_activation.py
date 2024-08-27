import numpy as np



class Custom_Activation_Layer:
    
    def __init__(
        self, activation_name='sigmoid_approx'
    ):
        
        if '_approx' in activation_name:
            self.activation_name = activation_name
        else:
            self.activation_name = activation_name + '_approx'
        
        return
    
    
    def vector_activation(self, _tensor, coefficients):
        
        y = coefficients[0]
        x_n = _tensor
        for coefficient in coefficients[1:]:
            y = y + coefficient * x_n
            x_n = x_n * _tensor
        
        return y
    
    
    def call(self, _tensor, layer, **kwargs):
        
        coefficients_all = {
        'sigmoid_approx': np.array([ 5.00000000e-01,  1.26189290e-01, -2.49658001e-17, -8.90963399e-04]),
        'relu_approx': np.array([0, 1, 1]),
        'square_approx': np.array([0, 0, 1]),
        'leaky_sigmoid_1_approx': np.array([ 5.00000000e-01,  1.26189290e-01+5e-3, -2.49658001e-17, -8.90963399e-04]),
        }
        coefficients = coefficients_all[self.activation_name]
        
        return [self.vector_activation(_tensor_, coefficients) for _tensor_ in _tensor]