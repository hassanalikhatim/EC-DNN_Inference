import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# import tensorflow_privacy
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

from utils_.general_utils import confirm_directory
from server_utils.model_architectures import mlp_model, cnn_model, attention_text



WEIGHT_DECAY = 1e-2

model_architectures = {
    'mlp': mlp_model,
    'cnn': cnn_model,
    'att': attention_text
}


class Keras_Model:
    def __init__(
        self, model_architecture,
        activation='sigmoid', n_layers=2,
        learning_rate=1e-4, weight_decay=WEIGHT_DECAY,
        data=None, path=None, data_name=None,
        embedding_depth=10,
        privacy=False,
        l2_norm_clip=1.5, noise_multiplier=1.3, delta=1e-5
    ):
        
        self.path = path
        self.data = data
        if data_name is None:
            self.data_name = self.data.data_name
        else:
            self.data_name = data_name
        
        self.model_architecture = model_architectures[model_architecture]
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.activation = activation
        self.apply_softmax = True
        
        self.embedding_depth = embedding_depth
        
        self.privacy = privacy
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        if self.privacy:
            self.apply_softmax = False
            self.weight_decay = 0

        self.prepare_model()
        
        return
    
    
    def prepare_model(self):
        
        self.model = self.model_architecture(
            self.data, weight_decay=self.weight_decay, 
            n_layers=self.n_layers,
            activation_name=self.activation,
            learning_rate=self.learning_rate,
            apply_softmax=self.apply_softmax,
            embedding_depth=self.embedding_depth
        )
        
        self.save_directory = self.path + self.data_name + '/'
        self.save_directory += str(self.n_layers) + '/models/'
        
        return
    
    
    def train(self, epochs=1, batch_size=None, patience=0):
        
        if self.privacy:
            optimizer = tensorflow_privacy.DPKerasSGDOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                              noise_multiplier=self.noise_multiplier,
                                                              num_microbatches=self.num_microbatches,
                                                              learning_rate=self.learning_rate)

            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.losses.Reduction.NONE
            )

            self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        early_stopping_monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )

        self.model.fit(
            self.data.x_train, self.data.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.data.x_test, self.data.y_test),
            callbacks=[early_stopping_monitor]
        )
        
        return
        
        
    def save(self, name):
        
        confirm_directory(self.save_directory)
        self.model.save_weights(self.save_directory + name + '.h5')
        
        return
        
        
    def load_or_train(self, name, epochs=1, batch_size=None, patience=1):

        if self.privacy:
            name += '_private'

            self.num_microbatches = batch_size
            if batch_size % self.num_microbatches != 0:
                raise ValueError('Batch size should be an integer multiple of the number of microbatches')
        
        if os.path.isfile(self.save_directory + name + '.h5'):
            self.model.load_weights(self.save_directory + name + '.h5')
            print('Loaded pretrained weights:', name)
        
        else:
            print('Model not found at: ', self.save_directory + name + '.h5')
            print('Training model from scratch.')
            self.train(epochs=epochs, batch_size=batch_size, patience=patience)
            self.save(name)

        # if self.privacy:
        #     self.get_differential_privacy(epochs=epochs, verbose=True)
            
        return
    

    def get_differential_privacy(self, epochs, verbose=False):

        self.epsilon, self.rdp = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=self.data.x_train.shape[0],
            batch_size=self.num_microbatches,
            noise_multiplier=self.noise_multiplier,
            epochs=epochs,
            delta=self.delta
        )

        if verbose:
            print('epsilon:', self.epsilon, ', rdp:', self.rdp)

        return
