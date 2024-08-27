import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from scripts_.configuration_values import *

from utils_.visual_utils import generate_broken_vertical_axis_plot, export_legend



markers = '123os^v<>'


def make_table_2(post_str, n_layers, scale_, load_path=None):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'cnn', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-4]
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN'
    }
    
    if not load_path:
        load_path = path
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    table_string = ''
    
    for model_architecture in model_architectures_:
        for model_activation in model_activations_:
            table_string += '{:s}$_{:s}$ & \n'.format(
                nice_names[model_architecture], nice_names[model_activation]
            )
            
            for d, dataset_name in enumerate(dataset_names_):
                
                for wd, weight_decay in enumerate(weight_decays_):
                    
                    col_name = dataset_name
                    col_name += '_' + model_architecture
                    col_name += '_' + model_activation
                    col_name += '_' + str(n_layers)
                    col_name += '_' + str(weight_decay)
                    col_name += '_' + str(scale_)
                    
                    if col_name in df_accuracies.columns:
                        app_acc, _, consistency, _, _, _ = df_accuracies[col_name].tolist()
                        table_string += '{:3.2f}~(\\enc{{{:3.2f}}})'.format(app_acc, consistency)
                    else:
                        table_string += '{--}'
                    
                    if d < (len(dataset_names_)-1) or wd < 1:
                        table_string += ' & '
                table_string += '\n'
            table_string += '\\\\\n'
        table_string += '\hline\n\n'
    
    return table_string


def make_figure_8(post_str, n_layers, scale_, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    colors = {
        'relu': 'blue', 'leaky_sigmoid_1': 'red', 'square': 'green'
    }
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    output_array = -1 * np.ones(
        (
            len(model_architectures_),
            len(dataset_names_),
            len(model_activations_),
            2,
            len(weight_decays_)
        )
    )
    
    for m_arc, model_architecture in enumerate(model_architectures_):
        figures = []
        the_pdf = PdfPages(save_path + model_architecture+'_'+str(n_layers)+'_regularization.pdf')
        
        for d, dataset_name in enumerate(dataset_names_):
            
            fig = plt.figure(figsize=(3.5, 2.1))
            for m_act, model_activation in enumerate(model_activations_):
                for wd, weight_decay in enumerate(weight_decays_):
                    
                    col_name = dataset_name
                    col_name += '_' + model_architecture
                    col_name += '_' + model_activation
                    col_name += '_' + str(n_layers)
                    col_name += '_' + str(weight_decay)
                    col_name += '_' + str(scale_)
                    
                    if col_name in df_accuracies.columns:
                        app_acc, _, consistency, _, en_loss, _ = df_accuracies[col_name].tolist()
                        output_array[m_arc, d, m_act, :, wd] = [consistency, en_loss]
                    
                plt.plot(
                    np.arange(len(weight_decays_)), output_array[m_arc, d, m_act, 0],
                    color=colors[model_activation], marker=markers[m_act],
                    label='{:s}$\'_{{{:s}\'}}$'.format(
                        nice_names[model_architecture], nice_names[model_activation]
                    )
                )
            
            plt.xticks(np.arange(len(weight_decays_)), weight_decays_)
            plt.xlabel('$L_2$ regularization strength')
            plt.ylabel('Consistency')
            # plt.title(nice_names[dataset_name])
            plt.tight_layout()
            plt.legend()
            figures.append(fig)
        
        for each_fig in figures:
            the_pdf.savefig(each_fig)
        the_pdf.close()
    
    return


def make_figure_9(post_str, n_layers, scale_, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    colors = {
        'relu': 'blue', 'leaky_sigmoid_1': 'red', 'square': 'green'
    }
    
    df_differences = pd.read_excel(
        load_path + 'differences' + post_str + '.xlsx',
        engine='openpyxl'
    )
    
    for m_arc, model_architecture in enumerate(model_architectures_):
        figures = []
        the_pdf = PdfPages(save_path + model_architecture+'_'+str(n_layers)+'_activation.pdf')
        
        for d, dataset_name in enumerate(dataset_names_):
            
            fig = plt.figure(figsize=(3.5, 2.1))
            for m_act, model_activation in enumerate(model_activations_):
                    
                col_name = dataset_name
                col_name += '_' + model_architecture
                col_name += '_' + model_activation
                col_name += '_' + str(n_layers)
                col_name += '_' + str(1e-4)
                col_name += '_' + str(scale_)
                
                if col_name in df_differences.columns:
                    output_array = np.array(df_differences[col_name].tolist())
                    output_array = output_array[np.where(output_array!=-1)[0]]
                        
                    plt.plot(
                        np.arange(len(output_array)), output_array,
                        color=colors[model_activation], marker=markers[m_act],
                        label='{:s}$\'_{{{:s}\'}}$'.format(
                            nice_names[model_architecture], nice_names[model_activation]
                        )
                    )
            
            x_ticks = ['In']
            for i in range(len(output_array)):
                x_ticks.append( 'FC$^'+str(i+1)+'$' )
                x_ticks.append( 'A$^'+str(i+1)+'$')
            x_ticks = x_ticks[:len(output_array)]
            plt.xticks(np.arange(len(output_array)), x_ticks)
            
            plt.xlabel('Layer Number')
            plt.ylabel('Loss')
            plt.yscale('log')
            # plt.title(nice_names[dataset_name])
            plt.tight_layout()
            plt.legend()
            figures.append(fig)
        
        for each_fig in figures:
            the_pdf.savefig(each_fig)
        the_pdf.close()
    
    return


def make_figure_10(post_str, scale_, weight_decay, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    model_depths_ = [1, 3, 5]
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    linestyles = ['dashed', 'solid']
    colors = {
        'relu': 'blue', 'leaky_sigmoid_1': 'red', 'square': 'green'
    }
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    output_array = -1 * np.ones(
        (
            len(model_architectures_),
            len(dataset_names_),
            len(model_activations_),
            2,
            len(model_depths_)
        )
    )
    
    for m_arc, model_architecture in enumerate(model_architectures_):
        figures = []
        the_pdf = PdfPages(save_path + model_architecture+'_depth'+'.pdf')
        
        for d, dataset_name in enumerate(dataset_names_):
            
            fig = plt.figure(figsize=(3.5, 2.1))
            for m_act, model_activation in enumerate(model_activations_):
                for nl, n_layers in enumerate(np.sort(model_depths_)):
                
                    col_name = dataset_name
                    col_name += '_' + model_architecture
                    col_name += '_' + model_activation
                    col_name += '_' + str(n_layers)
                    col_name += '_' + str(weight_decay)
                    col_name += '_' + str(scale_)
                    
                    if col_name in df_accuracies.columns:
                        _, _, consistency, _, en_loss, _ = df_accuracies[col_name].tolist()
                        output_array[m_arc, d, m_act, :, nl] = [consistency, en_loss]
                
                plt.plot(
                    np.arange(len(model_depths_)), output_array[m_arc, d, m_act, 0],
                    color=colors[model_activation], marker=markers[m_act],
                    label='{:s}$\'_{{{:s}\'}}$'.format(
                        nice_names[model_architecture], nice_names[model_activation]
                    )
                )
            
            plt.xticks(np.arange(len(model_depths_)), np.sort(model_depths_))
            plt.xlabel('Model depth ($d$)')
            plt.ylabel('Consistency')
            # plt.title(nice_names[dataset_name])
            plt.tight_layout()
            plt.legend(ncol=2)
            figures.append(fig)
        
        for each_fig in figures:
            the_pdf.savefig(each_fig)
        the_pdf.close()
    
    return


def make_figure_11(post_str, n_layers, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    model_depths_ = [1, 3, 5]
    
    dataset_name = 'Whatsapp_Misinformation'
    weight_decay = 0
    scales_ = np.array([22, 23, 24, 29])
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    linestyles = ['dashed', 'solid']
    colors = {
        'relu': 'blue', 'leaky_sigmoid_1': 'red', 'square': 'green'
    }
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    output_array = -1 * np.ones(
        (
            len(model_architectures_),
            len(model_activations_),
            len(scales_)
        )
    )
    
    for m_arc, model_architecture in enumerate(model_architectures_):
        fig = plt.figure(figsize=(3.5, 2.1))

        for m_act, model_activation in enumerate(model_activations_):
            for s, scale_ in enumerate(np.sort(scales_)):
                
                col_name = dataset_name
                col_name += '_' + model_architecture
                col_name += '_' + model_activation
                col_name += '_' + str(n_layers)
                col_name += '_' + str(weight_decay)
                col_name += '_' + str(scale_)
                
                if col_name in df_accuracies.columns:
                    _, _, consistency, _, en_loss, time = df_accuracies[col_name].tolist()
                    output_array[m_arc, m_act, s] = time
            
            plt.plot(
                np.sort(scales_), output_array[m_arc, m_act],
                color=colors[model_activation], 
                marker=markers[m_act],
                label='{:s}$\'_{{{:s}\'}}$'.format(
                    nice_names[model_architecture], nice_names[model_activation]
                )
            )
    
        # plt.xticks(np.arange(len(scales)), np.sort(scales_))
        plt.xlabel('Scale')
        plt.ylabel('Inference Time (sec)')
        # plt.title(nice_names[dataset_name])
        plt.tight_layout()
        plt.legend(ncol=2)
        
        plt.savefig(save_path + model_architecture+'_'+str(n_layers)+'_scale_time.pdf')
    
    return


def make_figure_12(post_str, n_layers, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    model_depths_ = [1, 3, 5]
    
    dataset_name = 'Whatsapp_Misinformation'
    weight_decay = 0
    scales_ = np.array([22, 23, 24, 29])
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    linestyles = ['dashed', 'solid']
    colors = {
        'mlp_relu': 'blue', 'mlp_leaky_sigmoid_1': 'red', 'mlp_square': 'green',
        'att_relu': 'black', 'att_leaky_sigmoid_1': 'orange', 'att_square': 'gray'
    }
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    output_array = -1 * np.ones(
        (
            len(model_architectures_),
            len(model_activations_),
            len(scales_)
        )
    )
    
    fig = plt.figure(figsize=(3.5, 2.5))
    for m_arc, model_architecture in enumerate(model_architectures_):
        for m_act, model_activation in enumerate(model_activations_):
            for s, scale_ in enumerate(np.sort(scales_)):
                
                col_name = dataset_name
                col_name += '_' + model_architecture
                col_name += '_' + model_activation
                col_name += '_' + str(n_layers)
                col_name += '_' + str(weight_decay)
                col_name += '_' + str(scale_)
                
                if col_name in df_accuracies.columns:
                    _, _, consistency, _, en_loss, time = df_accuracies[col_name].tolist()
                    output_array[m_arc, m_act, s] = consistency
            
            plt.plot(
                np.arange(len(scales_)), output_array[m_arc, m_act],
                color=colors[model_architecture+'_'+model_activation], 
                marker=markers[m_act],
                label='{:s}$\'_{{{:s}\'}}$'.format(
                    nice_names[model_architecture], nice_names[model_activation]
                )
            )
    
    plt.xticks(np.arange(len(scales_)), scales_)
    plt.xlabel('Scale ($s$)')
    plt.ylabel('Consistency')
    # plt.title(nice_names[dataset_name])
    plt.tight_layout()
    
    legend = plt.legend(
        ncol=len(model_activations) * len(model_architecture), 
        bbox_to_anchor=(2,1), loc='upper left',
        edgecolor='black'
    )
    export_legend(legend, filename=save_path+'scale_consistency_legend')
    
    plt.savefig(save_path + 'scale_consistency_'+str(n_layers)+'.pdf')
    
    return


def make_figure_11b(post_str, n_layers, load_path=path, save_path=path):
    
    dataset_names_ = ['cifar10', 'mnist', 'Kaggle_Fake_News', 'Whatsapp_Misinformation', 'IMDB']
    model_architectures_ = ['mlp', 'att']
    model_activations_ = ['relu', 'leaky_sigmoid_1', 'square']
    weight_decays_ = [0, 1e-5, 1e-4, 1e-3]
    model_depths_ = [1, 3, 5]
    
    dataset_name = 'Whatsapp_Misinformation'
    weight_decay = 0
    scales_ = np.array([22, 23, 24, 29])
    
    nice_names = {
        'relu': 'r', 'leaky_sigmoid_1': '\sigma', 'square': 'q',
        'mlp': 'MLP', 'cnn': 'CNN', 'att': 'SAN',
        'mnist': 'MNIST', 'cifar10': 'CIFAR-10',
        'Kaggle_Fake_News': 'Fake News', 'Whatsapp_Misinformation': 'Misinformation',
        'IMDB': 'IMDB'
    }
    linestyles = ['dashed', 'solid']
    colors = {
        'mlp_relu': 'blue', 'mlp_leaky_sigmoid_1': 'red', 'mlp_square': 'green',
        'att_relu': 'black', 'att_leaky_sigmoid_1': 'orange', 'att_square': 'gray'
    }
    
    df_accuracies = pd.read_excel(
        load_path + 'accuracies' + post_str + '.xlsx',
        engine='openpyxl'
    )
    output_array = -1 * np.ones(
        (
            len(model_architectures_),
            len(model_activations_),
            len(scales_)
        )
    )
    
    
    unified_dictionary_of_y_values = {}
    for m_arc, model_architecture in enumerate(model_architectures_):
        
        set_of_y_values = {}
        for m_act, model_activation in enumerate(model_activations_):
            for s, scale_ in enumerate(np.sort(scales_)):
                
                col_name = dataset_name
                col_name += '_' + model_architecture
                col_name += '_' + model_activation
                col_name += '_' + str(n_layers)
                col_name += '_' + str(weight_decay)
                col_name += '_' + str(scale_)
                
                if col_name in df_accuracies.columns:
                    _, _, consistency, _, en_loss, time = df_accuracies[col_name].tolist()
                    output_array[m_arc, m_act, s] = time
            
            label='{:s}$\'_{{{:s}\'}}$'.format(
                nice_names[model_architecture], nice_names[model_activation]
            )
            set_of_y_values[label] = {
                'values': output_array[m_arc, m_act],
                'color': colors[model_architecture+'_'+model_activation], 
                'marker': markers[m_act],
            }
        
        unified_dictionary_of_y_values[model_architecture] = set_of_y_values
        
    fig, legend_axis, additional_axis = generate_broken_vertical_axis_plot(
        np.sort(scales_),
        unified_dictionary_of_y_values['att'],
        unified_dictionary_of_y_values['mlp']
    )
    
    fig.supylabel('Inference Time (sec)')
    plt.xlabel('Scale ($s$)')
    plt.tight_layout()
    
    # legend = legend_axis.legend(
    #     ncol=len(set_of_y_values.keys()), 
    #     bbox_to_anchor=(2,1), loc='upper left',
    #     edgecolor='black'
    # )
    # export_legend(legend, filename=save_path+'scale_time_legend')
    
    plt.savefig(save_path+'scale_time_'+str(n_layers)+'.pdf')
    
    return