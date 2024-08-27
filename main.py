import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from scripts_.unencrypted_inference import main as main_unencrypted_inference
# from scripts_.encrypted_inference import main as main_encrypted_inference
# from scripts_.encrypted_inference_naive import main as main_encrypted_inference_naive



if __name__ == "__main__":
    
    main_unencrypted_inference()
    # main_encrypted_inference()
    