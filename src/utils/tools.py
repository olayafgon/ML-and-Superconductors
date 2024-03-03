import os
import sys
import shutil
import datetime 

sys.path.append('./../')


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