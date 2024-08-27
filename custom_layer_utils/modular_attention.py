import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


from custom_layer_utils.custom_functions import relu_approx, sigmoid_approx, square



class modular_Attention(layers.Layer):
    
    def __init__(
        self, units=32, activation_name='sigmoid', 
        weight_constraint=False,
        kernel_regularizer=None
    ):
        
        super(modular_Attention, self).__init__()
        
        self.units = units
        self.activation_name = activation_name
        self.weight_constraint = weight_constraint
        self.kernel_regularizer = kernel_regularizer
        
        self.activation_dictionary = {
            'relu': tf.keras.activations.relu,
            'sigmoid': tf.keras.activations.sigmoid,
            'square': square,
            'relu_approx': relu_approx,
            'sigmoid_approx': sigmoid_approx,
            'square_approx': square
        }
        
        return
    
    
    def build(self, input_shape):
        
        self.keys = self.add_weight(
            shape=(input_shape[2], self.units),
            name='keys',
            initializer="random_normal",
            regularizer=self.kernel_regularizer
        )
        
        self.values = self.add_weight(
            shape=(input_shape[2], self.units),
            name='values',
            initializer="random_normal",
            regularizer=self.kernel_regularizer
        )
        
        self.queries = self.add_weight(
            shape=(input_shape[2], self.units),
            name='queries',
            initializer="random_normal"
        )
        
        return
    
    
    def get_variables(self):
        return self.queries, self.keys, self.values
    

    def attention(self, inputs):
        
        inputs = inputs[:, :, :, 0]
        
        queries, keys, values = self.get_variables()
        
        queries_ = K.dot(inputs, queries)
        keys_ = K.dot(inputs, keys)
        values_ = K.dot(inputs, values)
        
        # print(queries_.shape, K.permute_dimensions(keys_[0], (1,0)).shape)
        queries_keys_ = self.activation_dictionary[self.activation_name](queries_ @ K.permute_dimensions(keys_, (0,2,1)))
        
        return [queries_keys_ @ values_, queries_, keys_, values_]
    
    
    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, -1, 1)
        return self.attention(inputs)[0]
    
      
    def check(self, inputs):
        inputs = tf.clip_by_value(inputs, -1, 1)
        return self.attention(inputs)