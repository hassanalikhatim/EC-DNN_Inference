# Hybrid configurations
pneumonia_np_balanced_values = {
    'epochs': 200,
    'batch_size': 512,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 500,
    'patience': 100,
    'learning_rate': 1e-4,
    'max_len': None
}
mri_values = {
    'epochs': 200,
    'batch_size': 32,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 50,
    'patience': 100,
    'learning_rate': 1e-5,
    'max_len': None
}
mnist_values = {
    'epochs': 1000,
    'batch_size': 512,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 100,
    'patience': 40,
    'learning_rate': 1e-3,
    'max_len': None
}
cifar10_values = {
    'epochs': 1000,
    'batch_size': 256,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 500,
    'patience': 100,
    'learning_rate': 1e-3,
    'max_len': None
}
kaggle_fakenews = {
   'epochs': 1000,
    'batch_size': 256,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 500,
    'patience': 40,
    'learning_rate': 1e-3,
    'max_len': 100
}


general_configurations = {
    'MRI': mri_values,
    'pneumonia_np_balanced': pneumonia_np_balanced_values,
    'mnist': mnist_values,
    'cifar10': cifar10_values,
    'fmnist': mnist_values,
    'IDC': mri_values,
    'Kaggle_Fake_News': kaggle_fakenews,
    'Whatsapp_Misinformation': kaggle_fakenews,
    'IMDB': kaggle_fakenews
}
abbreviations = {
    'sigmoid': 's',
    'relu': 'r',
    'square': 'q',
    'leaky_sigmoid': 'leaky_sigmoid',
    'leaky_sigmoid_1': 'leaky_sigmoid_1'
}


# Changeable configurations
# General
path = '_all_results/results_activity_regularization/'
force_save_results = False


# Data
SIZE = 80
dataset_folder = '../../_Datasets/'

dataset_names = []
dataset_names += ['cifar10']
dataset_names += ['mnist']
dataset_names += ['Kaggle_Fake_News']
dataset_names += ['Whatsapp_Misinformation']
dataset_names += ['IMDB']

n_samples = 39


# Model
privacy = False
embedding_depth = 10

model_activations = []
model_activations += ['relu']
# model_activations += ['sigmoid']
# model_activations += ['leaky_sigmoid']
model_activations += ['leaky_sigmoid_1']
model_activations += ['square']

model_architectures = []
model_architectures += ['mlp']
# model_architectures += ['cnn']
# model_architectures += ['att']

max_allowed_depth = 100
model_depths = [1, 2, 3, 4, 5]

weight_decays = []
weight_decays += [0, 1e-4]
# weight_decays += [0, 1e-5, 1e-4, 1e-3]


# Encryption
scales = [23]
depth = 6
degree = 15

