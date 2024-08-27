import os
import numpy as np


from utils_.results_storing import *



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_memory_usage():
    # Importing the library to measure RAM usage
    import psutil
    return psutil.virtual_memory()[2]


def add_element_to_dict(dict_, element_key, element):
    
    if element_key in dict_.keys():
        dict_[element_key] += [element]
    else:
        dict_[element_key] = [element]
        
    return dict_


def reverse(list_arr: list):
    _list_arr = list_arr[::-1]
    return _list_arr