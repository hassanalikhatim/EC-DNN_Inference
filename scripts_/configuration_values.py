import config



# Hybrid configurations
general_configurations = config.general_configurations
abbreviations = config.abbreviations

# General configurations
path = config.path
force_save_results = config.force_save_results

# Model configurations
embedding_depth = config.embedding_depth
model_activations = config.model_activations
model_architectures = config.model_architectures
model_depths = config.model_depths
max_allowed_depth = config.max_allowed_depth
weight_decays = config.weight_decays
privacy = config.privacy

# Data configurations
SIZE = config.SIZE
dataset_names = config.dataset_names
dataset_folder = config.dataset_folder
n_samples = config.n_samples

# Encryption configurations
scales = config.scales
depth = config.depth
degree = config.degree

# # Return dictionary
# # This dictionary can be used to return functions in multiprocessing
# global return_dict
# return_dict = {}