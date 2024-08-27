import shutil
import os


from utils_.general_utils import confirm_directory



names = ['ignore', '_all_results']

def nested_copy(directory, destination, names, ignore=True):
    confirm_directory(destination)
    
    all_dirs = os.listdir(directory)
    for dir_ in all_dirs:
        src_dir = directory + '/' + dir_
        dest_dir = destination + '/' + dir_
        
        if os.path.isdir(src_dir):
            if ignore:
                if not dir_ in names:
                    nested_copy(src_dir, dest_dir, names)
            else:
                if dir_ in names:
                    nested_copy(src_dir, dest_dir, names)
        else:
            print(dest_dir)
            shutil.copy(src_dir, dest_dir)
 

def smart_copy(names=names, ignore=True):
    print('Going to ignore these directories:', names)
    
    # path to source directory
    src_dir = '../Project15_Metaverse'
 
    # path to destination directory
    dest_dir = '../smart_copy/Project15_Metaverse'

    names = names
    nested_copy(src_dir, dest_dir, names, ignore=ignore)