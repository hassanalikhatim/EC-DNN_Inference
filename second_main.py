import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# gpus = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.set_visible_devices(gpus[3], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
# except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)



####################################
from scripts_.unencrypted_inference import main as main_unencrypted_inference
from scripts_.encrypted_inference import main as main_encrypted_inference
from scripts_.encrypted_inference_naive import main as main_encrypted_inference_naive



if __name__ == "__main__":
    
    # main_unencrypted_inference()
    main_encrypted_inference_naive()
    