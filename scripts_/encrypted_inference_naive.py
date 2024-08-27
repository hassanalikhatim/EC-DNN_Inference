import pandas as pd
import gc
import numpy as np

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
    kwargs, 
    return_dict
):
    
    dataset_configuration = kwargs['dataset_configuration']
    model_configuration = kwargs['model_configuration']
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
            ''.format(
                model_architecture, my_data.type
            )
        )
        return_dict['accuracies_dict'] = [-1, -1, -1]
        return_dict['differences_dict'] = [-1]*(2*n_layers+1)
        return
    
    # model with the non-polynomial activations
    my_model = Keras_Model(
        model_architecture, 
        n_layers=n_layers, activation=model_activation,
        data=my_data, weight_decay=weight_decay,
        path=model_path,
        embedding_depth=embedding_depth
    )
    my_model.load_or_train(
        model_name,
        epochs=epochs, batch_size=batch_size,
        patience=patience
    )
    _, org_acc = my_model.model.evaluate(
        my_data.x_test, my_data.y_test
    )
    
    # Defining the server and executing inference
    my_server = Debugging_Server(model=my_model)
    
    pre_compute_layers = []
    if my_data.type == 'nlp':
        pre_compute_layers = my_server.model.layers[:2]
    
    my_client = Debugging_Client(
        my_server, data_type=my_data.type, 
        pre_compute_layers=pre_compute_layers,
        depth=depth, scale=scale, degree=degree
    )
    
    try:
        outputs, decrypted_outputs = my_client.inference(my_data.x_test[test_samples])
    
        all_differences = compute_differences(outputs, decrypted_outputs)
        _, en_acc, consistency = compute_accuracies(
            outputs, decrypted_outputs, my_data.y_test[test_samples])
        
        return_dict['accuracies_dict'] = [org_acc, en_acc, consistency]
        return_dict['differences_dict'] = all_differences
    
    except  Exception as error:
        print('An error occured:', error)
        return_dict['accuracies_dict'] = [-1, -1, -1]
        return_dict['differences_dict'] = [-1]*(len(my_model.model.layers)-len(pre_compute_layers)-1)
        
    return


def main(orientation=1):
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    if orientation == 2:
        global dataset_names, model_architectures, model_activations, model_depths, weight_decays
        
        dataset_names = reverse(dataset_names)
        model_architectures = reverse(model_architectures)
        model_activations = reverse(model_activations)
        model_depths = reverse(model_depths)
        weight_decays = reverse(weight_decays)
    
    print(dataset_names)
    
    accuracies_dict = {}
    differences_dict = {}
    for dataset_name in dataset_names:
        col_name_data = dataset_name
        
        for model_architecture in model_architectures:
            col_name_architecture = col_name_data + '_' + model_architecture
            
            for model_activation in model_activations:
                col_name_activation = col_name_architecture + '_' + model_activation
                
                for n_layers in model_depths:
                    col_name_layers = col_name_activation + '_' + str(n_layers)
                    
                    for weight_decay in weight_decays:
                        col_name_decay = col_name_layers + '_' + str(weight_decay)
                    
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
                            args = (
                                {
                                    'dataset_configuration': dataset_configuration,
                                    'model_configuration': model_configuration,
                                    'path': path
                                },
                                return_dict,
                            )
                        )
                        process.start()
                        process.join()
                        
                        gc.collect()
                        
                        accuracies_dict[col_name_decay] = return_dict['accuracies_dict'].copy()
                        differences_dict[col_name_decay] = return_dict['differences_dict'].copy()
                        
                        print('\r{}_naive -:- '.format(col_name_decay))
                        print(
                            'Org Acc: {:.4f} | Enc Acc: {:.4f} | Consistency: {:.4f}'
                            ''.format(
                                accuracies_dict[col_name_decay][0], 
                                accuracies_dict[col_name_decay][1],
                                accuracies_dict[col_name_decay][2]
                            )
                        )
                        print(
                            'All differences: ',
                            ['{:.3f}'.format(d) for d in differences_dict[col_name_decay]]
                        )
    
    df_accuracies = pd.DataFrame.from_dict(accuracies_dict)
    df_accuracies.to_excel(path + 'accuracies_naive.xlsx', index=False)
    
    max_len = 0
    for key in differences_dict.keys():
        if len(differences_dict[key]) > max_len:
            max_len = len(differences_dict[key])
    for key in differences_dict.keys():
        differences_dict[key] += [-1] * (max_len - len(differences_dict[key]))
    df_differences = pd.DataFrame.from_dict(differences_dict)
    df_differences.to_excel(path + 'differences_naive.xlsx', index=False)
    
    return