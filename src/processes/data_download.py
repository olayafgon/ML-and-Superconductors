"""
This module defines a workflow to download data from Aflowlib (http://aflowlib.duke.edu).

Requires some configurations from config.py:
    - STRUCTURES
    - DATA_FOLDER_PATH
    - DATA_FILE_TYPE

Example:
    from processes import data_download
    downloader = data_download.data_download(run_results_path)
    downloader.data_download_workflow()
"""

import os
import sys
import subprocess
import requests
import urllib.request
import lzma
import shutil
import time
from utils import tools


sys.path.append('./../')
import config

class data_download:
    def __init__(self, run_results_path):
        """
        Initialize the data download module.

        Args:
            run_results_path (str): The path where the results and logs are stored.
        """
        self.run_results_path = run_results_path
        self.structures = config.STRUCTURES
        self.data_folder_path = config.DATA_FOLDER_PATH
        self.data_dowload_check = config.DATA_DOWNLOAD
        self.aflowlib_link = 'http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/'
        self.file_type = config.DATA_FILE_TYPE

    def get_names(self):
        """
        Create a folder in the data folder with files containing the names of all the materials available in Aflowlib for each bravais lattice.
        """
        materials_names_folder = os.path.join(self.data_folder_path, 'materials_names')
        tools.create_folder(materials_names_folder)
    
        for structure in self.structures:
            # Download info files from library
            response = requests.get(self.aflowlib_link + structure)
            lines = response.text.split("\n")

            # Modify files to get names
            modified_lines = []
            for line in lines:
                if "_ICSD" in line:
                    parts = line.split("<")
                    for part in parts:
                        if "\">" in part and "aflowlib" not in part:
                            name = part.split(">")[1]
                            modified_lines.append(name + "\n")

            # Save
            modified_file_path = os.path.join(materials_names_folder, f'{structure}_names.txt')
            with open(modified_file_path, "w") as f_out:
                for line in modified_lines:
                    f_out.write(line)

    def download_compressed_data(self):
        """
        Download compressed data files from Aflowlib.
        """
        materials_names_folder = os.path.join(self.data_folder_path, "materials_names")
        data_compressed_folder = os.path.join(self.data_folder_path, "data_compressed")
        error_file_path = os.path.join(data_compressed_folder, "00_Errors.txt")

        for structure_name in self.structures:
            print(f'    - Downloading data for {structure_name}...')
            start_time_structure = time.time()

            output_directory = os.path.join(data_compressed_folder, structure_name)
            tools.create_folder(output_directory)

            # For each structure
            entry_file = os.path.join(materials_names_folder, f'{structure_name}_names.txt')
            with open(entry_file, "r") as file:
                for line in file:
                    line = line.strip()
                    
                    j = 0
                    for aa in line:
                        if aa == ".":
                            break
                        j += 1
                    aa = structure_name
                    bb = line[0:j]
                    
                    if self.file_type == 'DOSCAR.static.xz':
                        url = f'{self.aflowlib_link}{aa}/{bb}/{self.file_type}'
                    if self.file_type == '_dosdata.json.xz':
                        file_name = bb + self.file_type
                        file_name_cleaned = tools.clean_path(file_name) # Avoid errors
                        url = f'{self.aflowlib_link}{aa}/{bb}/{file_name_cleaned}'
                    file_path = os.path.join(output_directory, f"{aa}_{bb}.xz")

                    try:
                        urllib.request.urlretrieve(url, file_path)
                    except urllib.error.HTTPError as e:
                        with open(error_file_path, 'a') as error_file:
                            error_file.write(f'Error in {url}: {str(e)}\n')
                
            elapsed_time_structure = time.time() - start_time_structure
            print(f'     Downloaded all files for {structure_name} ({elapsed_time_structure:.2f} s)')

    def decompress_data(self):
        """
        Decompresses files from the source directory to the destination directory.
        """
        # Path to compressed files
        source_folder_path = os.path.join(self.data_folder_path, "data_compressed")
        # Path decompress the files
        destination_folder_path = os.path.join(self.data_folder_path, "data_raw")
        # Path to the .txt file to save names of files that cannot be decompressed
        error_file_path = os.path.join(destination_folder_path, "00_Errors.txt")

        if not os.path.exists(source_folder_path):
            print(f'Error: The source directory "{source_folder_path}" does not exist.')
            exit()
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        speed_limit = 2 * 500 * 1024 * 1024 # 2*500Mb/s
        for folder, subfolder, files in os.walk(source_folder_path):
            print(f'    - Decompressing data for {folder}...')
            start_time = time.time()
            for compress_file in files:
                if compress_file.endswith('.xz'):
                    # Constructs input and output paths
                    in_path = os.path.join(folder, compress_file)
                    out_folder = folder.replace(source_folder_path, '').lstrip(os.path.sep)
                    if self.file_type == 'DOSCAR.static.xz':
                        ruta_salida = os.path.join(destination_folder_path, out_folder, compress_file.replace('.xz', '.txt'))
                    if self.file_type == '_dosdata.json.xz':
                        ruta_salida = os.path.join(destination_folder_path, out_folder, compress_file.replace('.xz', '.json'))
                    
                    os.makedirs(os.path.join(destination_folder_path, out_folder), exist_ok=True)
                    try:
                        with lzma.open(in_path, 'rb') as f_in, open(ruta_salida, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out, length=speed_limit)
                    except Exception as e:
                        with open(error_file_path, 'a') as error_file:
                            error_file.write(f'Error in {in_path}: {str(e)}\n')
                        print(f'· Error in {in_path}: {str(e)}')
                        continue  

            elapsed_time = time.time() - start_time
            print(f'     Decompressed all files for {folder} ({elapsed_time:.2f} s)')

    def data_download_workflow(self):
        """
        Manages the data download workflow.

        This function checks the existence of required folders and guides the user through the data download process.
        It prompts the user for input based on the existing folders and performs appropriate actions accordingly.
        """

        # Construct paths to required folders
        materials_names_folder = os.path.join(self.data_folder_path, "materials_names")
        data_compressed_folder = os.path.join(self.data_folder_path, "data_compressed")
        data_raw_folder = os.path.join(self.data_folder_path, "data_raw")

        # Complete workflow to download data
        def whole_download_workflow():
            """
            Perform the complete workflow to download data.

            This function guides the user through the entire data download process.
            It prompts the user for confirmation before starting the download process and performs the following steps:
                1. Retrieve names of materials available in Aflowlib.
                2. Download compressed data files from Aflowlib.
                3. Decompress downloaded data files.
            """
            confirm = input('WARNING: You are about to start downloading files, this can take up to several hours. Are you sure you want to continue? (y/n): ')
            if confirm.lower() == 'y':
                print("  · Starting download. Please, be patient.")
                self.get_names()
                self.download_compressed_data()
                self.decompress_data()
                return
            elif confirm.lower() == 'n':
                print("  · File download cancelled. Exiting the program...")
                sys.exit()
            else:
                print("  · Invalid entry. Exiting the program...")
                sys.exit()

        # Check existence of required folders
                
        # All 3 folders exist
        if os.path.exists(materials_names_folder) and os.path.exists(data_compressed_folder) and os.path.exists(data_raw_folder):
            confirm = input("WARNING: All required folders already exist. Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
            if confirm.lower() != 'y':
                confirm = input("  · Do you want to use existing data? (y/n): ")
                if confirm.lower() != 'y':
                    print("  · Data download cancelled. Exiting the program...")
                    sys.exit()
                else:
                    print(f"  · Using existing data in {self.data_folder_path}")
                    return
            else:
                whole_download_workflow()

        # Exist materials_names and data_compressed
        elif os.path.exists(materials_names_folder) and os.path.exists(data_compressed_folder):
            confirm = input("  · Materials names and data compressed folder exists. Do you want to decompress and use it? (y/n): ")
            if confirm.lower() == 'y':
                self.decompress_data()
            else:
                confirm = input("  · Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
                if confirm.lower() != 'y':
                    print("  · Data download cancelled. Exiting the program...")
                    sys.exit()
                else:
                    whole_download_workflow()

        # Exist materials_names and data_raw
        elif os.path.exists(materials_names_folder) and os.path.exists(data_raw_folder):
            confirm = input("  · Materials names and data raw folder exists. Downloading new data may overwrite existing files. Do you want to continue downloading the data? (y/n): ")
            if confirm.lower() != 'y':
                confirm = input("  · Do you want to use existing data? (y/n): ")
                if confirm.lower() != 'y':
                    print("  · Data download cancelled. Exiting the program...")
                    sys.exit()
                else:
                    print(f"  · Using existing data in {self.data_folder_path}")
                    return
            else:
                whole_download_workflow()
        
        # Exist data_compressed and data_raw
        elif os.path.exists(data_compressed_folder) and os.path.exists(data_raw_folder):
            confirm = input("  · Data compressed and data raw folder exists. Downloading new data may overwrite existing files. What do you want to do? (continue/use compressed data/use raw data): ")
            if confirm.lower() == 'use compressed data':
                self.decompress_data()
            elif confirm.lower() == 'use raw data':
                print(f"  · Using existing data in {self.data_folder_path}")
                return
            elif confirm.lower() == 'continue':
                whole_download_workflow()
            else:
                print("  · Invalid entry. Exiting the program...")
                sys.exit()

        # Only exists materials_names
        elif os.path.exists(materials_names_folder):
            confirm = input("  · Materials names folder already exists. Do you want to update the materials list or keep the existing one? (update/keep): ")
            if confirm.lower() == 'update':
                self.get_names()
            self.download_compressed_data()
            self.decompress_data()
            return

        # Only exists data_compressed
        elif os.path.exists(data_compressed_folder):
            confirm = input("  · Data compressed folder exists. Do you want to decompress and use it? (y/n): ")
            if confirm.lower() == 'y':
                self.decompress_data()
            else:
                confirm = input("  · Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
                if confirm.lower() != 'y':
                    print("  · Data download cancelled. Exiting the program...")
                    sys.exit()
                else:
                    whole_download_workflow()

        # Only exists data_raw
        elif os.path.exists(data_raw_folder):
            confirm = input("  · Data raw folder exists. Downloading new data may overwrite existing files. Do you want to continue downloading the data? (y/n): ")
            if confirm.lower() != 'y':
                confirm = input("  · Do you want to use existing data? (y/n): ")
                if confirm.lower() != 'y':
                    print("  · Data download cancelled. Exiting the program...")
                    sys.exit()
                else:
                    print(f"  · Using existing data in {self.data_folder_path}")
                    return
            else:
                whole_download_workflow()
        
        # None exist
        else:
            whole_download_workflow()
