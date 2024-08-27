from tensorflow.keras import utils
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from sklearn.utils import shuffle


from ..dataset import Dataset



class Med_MNIST_Dataset(Dataset):
    
    def __init__(
        self,
        preferred_size=None,
        **kwargs
    ):
        
        Dataset.__init__(
            self,
            data_name='Med_MNIST',
            preferred_size=preferred_size
        )
        
        return
    
    
    def prepare_data(
        self
    ):
        
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], x_train.shape[2], -1))
        
        num_classes = int(np.max(y_train) + 1)
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test
    
    
    def load_data(
        self
    ):
        
        x_train = np.load('../Datasets/'+self.data_name+'/x_train.npy')
        y_train = np.load('../Datasets/'+self.data_name+'/y_train.npy')
        
        x_test = np.load('../Datasets/'+self.data_name+'/x_test.npy')
        y_test = np.load('../Datasets/'+self.data_name+'/y_test.npy')
        
        return (x_train, y_train), (x_test, y_test)
    
    
    def get_class_names(
        self
    ):
        
        data_class_names = os.listdir('../Datasets/'+self.data_name+'/train/')
        
        return data_class_names
    
    