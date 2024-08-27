import pandas as pd
import gc
import numpy as np
import os

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from utils_.general_utils import compute_accuracies, compute_differences, reverse

from server_utils.model import Keras_Model

from data_utils.dataset_cards.common_dataset import Common_Dataset
from data_utils.dataset_cards.mri import MRI_Dataset
from data_utils.dataset_cards.idc import IDC_Dataset
from data_utils.nlp_dataset import NLP_Dataset
from data_utils.dataset_cards.kaggle_fakenews import Kaggle_Fakenews
from data_utils.dataset_cards.whatsapp_misinformation import Whatsapp_Misinformation

from server_utils.debugging_server import Debugging_Server
from client_utils.debugging_client import Debugging_Client

from scripts_.configuration_values import *



Datasets = {
    'mnist': Common_Dataset,
    'fmnist': Common_Dataset,
    'cifar10': Common_Dataset,
    'MRI': MRI_Dataset,
    'IDC': IDC_Dataset,
    'Kaggle_Fake_News': Kaggle_Fakenews,
    'Whatsapp_Misinformation': Whatsapp_Misinformation,
    'IMDB': NLP_Dataset
}


def sub_main(
    kwargs
):
    
    dataset_configuration = kwargs['dataset_configuration']
    model_configuration = kwargs['model_configuration']
    scale = kwargs['scale']
    path = kwargs['path']
    
    
    dataset_name = dataset_configuration['dataset_name']
    max_len = dataset_configuration['max_len']
    epochs = dataset_configuration['epochs']
    batch_size = dataset_configuration['batch_size']
    patience = dataset_configuration['patience']
    
    model_architecture = model_configuration['model_architecture']
    n_layers = model_configuration['n_layers']
    model_activation = model_configuration['model_activation']
    weight_decay = model_configuration['weight_decay']
    
    my_data = Datasets[dataset_name](
        data_name=dataset_name,
        max_len=max_len,
        dataset_folder=dataset_folder
    )
    my_data.renew_data()
    # Prepare random test samples to evaluate the models
    test_samples = np.random.choice(np.arange(len(my_data.x_test)), size=n_samples, replace=False)

    # set automatic model path and model name
    model_path = path + 'model_' + model_architecture + '/' 
    model_name = abbreviations[model_activation]
    model_name += '_' + dataset_name
    model_name += '_' + str(weight_decay)
    if my_data.type == 'nlp':
        # for NLP tasks, we include the embedding depth in the model name so that we can train
        # different models for different embedding depths (if required).
        model_name += '_' + str(embedding_depth)
    elif model_architecture=='att':
        # We do not train self-attention networks for CV tasks
        print(
            'Skipping because the model architecture is {} while the dataset type is {}.'
            ''.format(model_architecture, my_data.type)
        )
        return
    
    # model with the approximate polynomial activations
    approx_model = Keras_Model(
        model_architecture, 
        n_layers=n_layers, activation=model_activation+'_approx',
        data=my_data, weight_decay=weight_decay,
        path=model_path,
        embedding_depth=embedding_depth
    )
    approx_model.load_or_train(
        'approx/' + model_name,
        epochs=epochs, batch_size=batch_size,
        patience=patience
    )
    app_loss, app_acc = approx_model.model.evaluate(
        my_data.x_test, my_data.y_test
    )
    
    # Defining the server and executing inference
    my_server = Debugging_Server(model=approx_model)
    
    pre_compute_layers = []
    if my_data.type == 'nlp':
        pre_compute_layers = my_server.model.layers[:2]
    
    my_client = Debugging_Client(
        my_server, data_type=my_data.type, 
        pre_compute_layers=pre_compute_layers,
        depth=depth, scale=scale, degree=degree
    )
    
    # loading results file
    column_name = dataset_name
    column_name += '_' + model_architecture
    column_name += '_' + model_activation
    column_name += '_' + str(n_layers)
    column_name += '_' + str(weight_decay)
    column_name += '_' + str(scale)
    
    accuracies_filename = 'accuracies_' + str(len(test_samples))
    differences_filename = 'differences_' + str(len(test_samples))
    if os.path.isfile(path + accuracies_filename + '.xlsx'):
        df_accuracies = pd.read_excel(
            path + accuracies_filename + '.xlsx', 
            engine='openpyxl'
        )
        df_differences = pd.read_excel(
            path + differences_filename + '.xlsx',
            engine='openpyxl'
        )
    else:
        print('\n\nCreating new dataframes.\n\n')
        df_accuracies = pd.DataFrame()
        df_differences = pd.DataFrame()
    
    assert len(df_differences.columns) == len(df_accuracies.columns)
    
    if (column_name not in df_differences.columns) or force_save_results:
        try:
            outputs, decrypted_outputs, times = my_client.inference(my_data.x_test[test_samples])
        
            all_differences = compute_differences(outputs, decrypted_outputs)
            (_, _), (en_loss, en_acc), consistency = compute_accuracies(
                outputs, decrypted_outputs, my_data.y_test[test_samples]
            )
            avg_time = np.mean(times)
            
        except  Exception as error:
            print('An error occured:', error)
            en_loss, en_acc, consistency, avg_time = -1, -1, -1, -1
            all_differences = [-1]*max_allowed_depth
            
        print('\n\n' + column_name + ' -:- ')
        print(
            'Org Acc: {:.4f} | Enc Acc: {:.4f} | Consistency: {:.4f}'
            ''.format(app_acc, en_acc, consistency)
        )
        print('All differences: ', [float('{:.3f}'.format(d)) for d in all_differences])
        print('\n\n')
        
        df_accuracies[column_name] = [app_acc, en_acc, consistency, app_loss, en_loss, avg_time]
        df_differences[column_name] = all_differences + [-1]*(max_allowed_depth - len(all_differences))
        
        df_accuracies.to_excel(path + accuracies_filename + '.xlsx', index=False)
        df_differences.to_excel(path + differences_filename + '.xlsx', index=False)
        
    else:
        print('\nResults already stored. Skipping {}'.format(column_name))
        
    return


def main(orientation=1):
    
    if orientation == 2:
        global dataset_names, model_architectures, model_activations, model_depths, weight_decays
        
        dataset_names = reverse(dataset_names)
        model_architectures = reverse(model_architectures)
        model_activations = reverse(model_activations)
        model_depths = reverse(model_depths)
        weight_decays = reverse(weight_decays)
    
    print(dataset_names)
    
    for dataset_name in dataset_names:
        for model_architecture in model_architectures:
            for model_activation in model_activations:
                for n_layers in model_depths:
                    for weight_decay in weight_decays:
                        for scale in scales:
                            
                            dataset_configuration = {
                                'dataset_name': dataset_name,
                                'max_len': general_configurations[dataset_name]['max_len'],
                                'epochs': general_configurations[dataset_name]['epochs'],
                                'batch_size': general_configurations[dataset_name]['batch_size'],
                                'patience': general_configurations[dataset_name]['patience']
                            }
                            
                            model_configuration = {
                                'model_architecture': model_architecture,
                                'model_activation': model_activation,
                                'n_layers': n_layers,
                                'weight_decay': weight_decay
                            }
                            
                            process = multiprocessing.Process(
                                target = sub_main,
                                args = ({
                                    'dataset_configuration': dataset_configuration,
                                    'model_configuration': model_configuration,
                                    'scale': scale,
                                    'path': path
                                },)
                            )
                            process.start()
                            process.join()
                            
                            gc.collect()

    return