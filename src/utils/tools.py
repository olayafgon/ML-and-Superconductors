import os
import sys
import shutil
import datetime 
import re
import matplotlib.pyplot as plt

sys.path.append('./../')
import config

def log(text, save_path, printing = True, create = False):
    """
    Logs the provided text to a file, creating the file if it does not exist, and optionally prints it to the console.

    Args:
        text (str): The text to be logged.
        save_path (str): The path where the log file will be created or updated.
        printing (bool, optional, default=True): If True, the log message is also printed to the console.
        create (bool, optional, default=False): If True, a new log file is created; otherwise, the log is appended to an existing file.

    Returns:
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
    Logs the provided text to the main log file and optionally prints it to the console.

    Args:
        text (str): The text to be logged.
        save_path (str): The path where the log file will be created or updated.
        printing (bool, optional, default=True): If True, the log message is also printed to the console.
        create (bool, optional, default=False): If True, a new main log file is created; otherwise, the log is appended to an existing file.

    Returns:
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
    """
    Creates a folder in a specific path, optionally deleting any existing folder at that location.

    Args:
        path (str): The path to the folder that needs to be created or checked.
        delete_old (bool, default: False): If True and the folder already exists, it will be deleted before creating a new one.

    Returns:
        None
    """
    if (delete_old == True) & os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

def create_run_results_folder():
    """
    Creates the results folder for the current run and returns its path.

    The folder name is based on the current date and time. A copy of the config.py file is also copied into the results folder.

    Returns:
        str: Path to the created results folder.
    """
    all_results_path = config.RESULTS_FOLDER
    run_results_path = os.path.join(all_results_path, str(datetime.datetime.now().strftime('%Y%m%d_%H_%M')))
    create_folder(all_results_path)
    create_folder(run_results_path)
    shutil.copy('../config.py', run_results_path)
    log_main(f'MODULE: Results folder created: {run_results_path}', save_path=run_results_path)
    return run_results_path

def copy_file(source_path, destination_path):
    """
    Copies a file from one directory to another.

    Args:
        source_path (str): Path of the original file.
        destination_path (str): Destination path.

    Returns:
        None
    """
    shutil.copy(source_path, destination_path)
    print(f"File copied from {source_path} to {destination_path}")

def clean_path(path):
    """
    Removes invalid characters from a path string, replacing them with underscores.

    Args:
        path (str): The path string to clean.

    Returns:
        str: The cleaned path string.
    """
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned_path = re.sub(invalid_chars, '_', path)
    return cleaned_path

def save_plot(results_folder_path, figure_name):
    """
    Saves a Matplotlib plot to a file in the specified results folder.

    Args:
        results_folder_path (str): The path to the results folder.
        figure_name (str): The desired name for the saved plot file.

    Returns:
        None
    """
    figure_path = os.path.join(results_folder_path, figure_name)
    plt.savefig(figure_path, bbox_inches='tight')

def write_to_report(report_file, text):
    """Writes text to the report file."""
    with open(report_file, 'a') as f:
        f.write(text)

