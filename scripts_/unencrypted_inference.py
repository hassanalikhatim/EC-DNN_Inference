import os
import pandas as pd
import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


from utils_.general_utils import add_element_to_dict, compute_accuracies, compute_differences

from server_utils.model import Keras_Model

from data_utils.dataset_cards.common_dataset import Common_Dataset
from data_utils.dataset_cards.mri import MRI_Dataset
from data_utils.dataset_cards.idc import IDC_Dataset
from data_utils.nlp_dataset import NLP_Dataset
from data_utils.dataset_cards.kaggle_fakenews import Kaggle_Fakenews
from data_utils.dataset_cards.whatsapp_misinformation import Whatsapp_Misinformation

from utils_.general_utils import reverse

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
    kwargs: dict
):
    """
    Inputs:
        {kwargs}: is a dictionary of multiple arguments. This module expects three arguments
            to be present in the dictionary.
            {kwargs['dataset_configuration']} is the dataset configuration,
            {kwargs['model_configuration']} is the model configuration,
            {kwargs['path']} is the path to store the results which include models and excel file,
            {kwargs['save_col']} will the column name of the excel file
            
    Outputs:
        returns nothing
    """
    
    dataset_configuration = kwargs['dataset_configuration']
    model_configuration = kwargs['model_configuration']
    path = kwargs['path']
    col_name = kwargs['col_name']
    
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
            'Skipping because the model architecture is {} while the dataset type is {}.'.format(
                model_architecture, my_data.type)
        )
        return -1, -1, -1, -1
    
    # actual model with non-polynomial activations
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
    org_loss, org_acc = my_model.model.evaluate(my_data.x_test[:n_samples], my_data.y_test[:n_samples])
    
    # model with the approximate polynomial activations
    my_server = Keras_Model(
        model_architecture, 
        n_layers=n_layers, activation=model_activation+'_approx',
        data=my_data, weight_decay=weight_decay,
        path=model_path,
        embedding_depth=embedding_depth
    )
    my_server.load_or_train(
        'approx/' + model_name,
        epochs=epochs, batch_size=batch_size,
        patience=patience
    )
    app_loss, app_acc = my_server.model.evaluate(my_data.x_test[:n_samples], my_data.y_test[:n_samples])
    
    # save results in excel file
    loss_accuracies_df_name = 'unencrypted_losses_accuracies'
    if os.path.isfile(path + loss_accuracies_df_name + '.xlsx'):
        df_accuracies = pd.read_excel(path + loss_accuracies_df_name + '.xlsx', engine='openpyxl')
    else:
        print('*******THIS IS BAD********')
        df_accuracies = pd.DataFrame()
    df_accuracies[col_name] = [org_loss, org_acc, app_loss, app_acc]
    df_accuracies.to_excel(path + loss_accuracies_df_name + '.xlsx', index=False)
    
    print('\n' + col_name + ' -:- ', end='')
    print('Unencrypted Loss: {:.4f} | Approximate Loss: {:.4f}'.format(org_loss, app_loss))
    
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
                            target=sub_main,
                            args=({
                                'dataset_configuration': dataset_configuration,
                                'model_configuration': model_configuration,
                                'path': path,
                                'col_name': col_name_decay
                            },)
                        )
                        process.start()
                        process.join()
                        
                        gc.collect()
    
    return