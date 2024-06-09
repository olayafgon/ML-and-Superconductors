import os
import sys
import shutil
import datetime 
import re

sys.path.append('./../')
import config

def log(text, save_path, printing = True, create = False):
    """
    Logs the provided text to a file, creating file if it does not exist, and optionally prints it.

    ARGS:
        - text (str): The text to be logged.
        - save_path (str): The path where the log file will be created or updated.
        - printing (bool, optional, default=True): If True, the log message is also printed to the console.
        - create (bool, optional, default=False): If True, a new log file is created; otherwise, the log is appended to an existing file.

    RETURNS:
        None
    """
    name = save_path.split('/')[-2]
    log_path = os.path.join(save_path, 'log_'+name+'.txt')

    if create==True:
        with open(log_path, 'w', encoding='utf-8') as file:
            file.write(text+' \n') 
    else:
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if printing == True:
            print(str(dt)+':    '+text)
        with open(log_path, 'a', encoding='utf-8') as file:
            file.write(str(dt)+': '+text+' \n')

def log_main(text, save_path, printing = True, create = False):
    """
    Logs the provided text to the main log file and optionally prints it.

    ARGS:
        - text (str): The text to be logged.
        - save_path (str): The path where the log file will be created or updated.
        - printing (bool, optional, default=True): If True, the log message is also printed to the console.
        - create (bool, optional, default=False): If True, a new main log file is created; otherwise, the log is appended to an existing file.

    RETURNS:
        None
    """
    log_path = os.path.join(save_path, 'log.txt')

    if create == True:
        with open(log_path, 'w', encoding='utf-8') as file:
            file.write(text+' \n') 
    else:
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if printing == True:
            print(str(dt)+': '+text)
        with open(log_path, 'a', encoding='utf-8') as file:
            file.write(str(dt)+': '+text+' \n')

def create_folder(path, delete_old = False):
    '''
    Creates a folder in an specific path.  

    ARGS:
        - path (str): The path to the folder that needs to be created or checked.
        - delete_old (bool, default: False) : If True and the folder already exists, it will be deleted before creating a new one.

    RETURNS:
        None
    '''
    if (delete_old == True) & os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def create_run_results_folder():
    '''
    Creates the results folder for the run and return its path.

    RETURNS:
        - str: Path to the created results folder.
    '''
    all_results_path = config.RESULTS_FOLDER
    run_results_path = os.path.join(all_results_path, str(datetime.datetime.now().strftime('%Y%m%d_%H_%M')))
    create_folder(all_results_path)
    create_folder(run_results_path)
    shutil.copy('../config.py', run_results_path)
    log_main(f'MODULE: Results folder created: {run_results_path}', save_path=run_results_path)
    return run_results_path

def copy_file(source_path, destination_path):
    """
    This function copy a file from one directory to another

    ARGS:
        - source_path (str): Path of original file.
        - destination_path (str): Destination path.
    
    RETURNS:
        None
    """
    shutil.copy(source_path, destination_path)
    print(f"File copied from {source_path} to {destination_path}")

def clean_path(path):
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned_path = re.sub(invalid_chars, '_', path)
    return cleaned_path