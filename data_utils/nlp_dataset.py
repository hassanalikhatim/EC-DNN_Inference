import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf


from data_utils.dataset import Dataset



class NLP_Dataset(Dataset):
    
    def __init__(
        self,
        data_name='IMDB',
        dataset_folder='../../_Datasets/',
        max_len=100, test_ratio=0.17,
        **kwargs
    ):
        
        Dataset.__init__(
            self,
            data_name=data_name
        )
        
        self.dataset_folder = dataset_folder
        self.test_ratio = test_ratio
        self.max_len = max_len
        
        self.type = 'nlp'
        
        return
    
    
    def prepare_data(
        self
    ):
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.prepare_nlp_things()
        
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    
    def prepare_nlp_things(self):
        
        self.prepare_tokenizer(self.x_train, self.x_test)
        self.x_train, self.x_test = self.get_tokenized(self.x_train, self.x_test)
        self.x_train, self.y_train = self.make_even(self.x_train, self.y_train)
        
        return
    
    
    def load_data(self):
        
        train_df = pd.read_csv(self.dataset_folder + 'IMDB/train.csv')
        ytrain = train_df['label'].tolist()
        x_train = train_df['content'].tolist()
        
        test_df = pd.read_csv(self.dataset_folder + 'IMDB/test.csv')
        ytest = test_df['label'].tolist()
        x_test = test_df['content'].tolist()
        
        num_classes = len(list(set(ytrain)))
        
        y_train = tf.keras.utils.to_categorical(np.array(ytrain)-1, num_classes)
        y_test = tf.keras.utils.to_categorical(np.array(ytest)-1, num_classes)
        
        return x_train, y_train, x_test, y_test
    
    
    def prepare_tokenizer(self, x_train, x_test):
        
        #Tokenize the sentences
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(list(x_train)+list(x_test))
        
        self.size_of_vocabulary = len(self.tokenizer.word_index) + 1 #+1 for padding
        
        print("The size of the vocabulary is: ", self.size_of_vocabulary)
        
        return self.tokenizer, self.size_of_vocabulary

 
    def get_tokenized(self, x_train, x_test):
        
        #converting text into integer sequences
        x_tr_seq  = self.tokenizer.texts_to_sequences(x_train)
        x_val_seq = self.tokenizer.texts_to_sequences(x_test)
        
        #padding to prepare sequences of same length
        x_tr_seq  = pad_sequences(x_tr_seq, maxlen=self.max_len, truncating='post')
        x_val_seq = pad_sequences(x_val_seq, maxlen=self.max_len, truncating='post')
        
        return x_tr_seq, x_val_seq


    def make_even(self, x_tr_seq, y_train):
        
        train_labels = np.argmax(y_train, axis=1)
        
        l0 = len(x_tr_seq[np.where(train_labels==0)])
        l1 = len(x_tr_seq[np.where(train_labels==1)])
        l = min(l0, l1)
        
        x_tr_seq_0 = x_tr_seq[np.where(train_labels==0)][:l]
        x_tr_seq_1 = x_tr_seq[np.where(train_labels==1)][:l]
        y_train_0 = y_train[np.where(train_labels==0)][:l]
        y_train_1 = y_train[np.where(train_labels==1)][:l]

        x_tr_seq = np.append(x_tr_seq_0, x_tr_seq_1, axis=0)
        y_train = np.append(y_train_0, y_train_1, axis=0)

        x_tr_seq, y_train = shuffle(x_tr_seq, y_train)

        return x_tr_seq, y_train