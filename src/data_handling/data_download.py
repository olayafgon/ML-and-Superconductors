import os
import time
import logging
import lzma
import shutil
import requests
import urllib
import sys

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
        tools.log_main(f'MODULE: data_download...', save_path=self.run_results_path)

    def get_names(self):
        """Creates a folder with files containing material names for each Bravais lattice."""
        materials_names_folder = os.path.join(self.data_folder_path, 'materials_names')
        tools.create_folder(materials_names_folder)
    
        for structure in self.structures:
            response = requests.get(self.aflowlib_link + structure)
            lines = response.text.split("\n")
            modified_lines = []
            for line in lines:
                if "_ICSD" in line:
                    parts = line.split("<")
                    for part in parts:
                        if "\">" in part and "aflowlib" not in part:
                            name = part.split(">")[1]
                            modified_lines.append(name + "\n")
            modified_file_path = os.path.join(materials_names_folder, f'{structure}_names.txt')
            with open(modified_file_path, "w") as f_out:
                for line in modified_lines:
                    f_out.write(line)

        tools.log_main(f'Starting data dowload', save_path=self.run_results_path)

    def download_compressed_data(self):
        """Downloads compressed data files from Aflowlib."""

        materials_names_folder = os.path.join(self.data_folder_path, "materials_names")
        data_compressed_folder = os.path.join(self.data_folder_path, "data_compressed")

        for structure_name in self.structures:
            print(f'  - Downloading data for {structure_name}...')
            start_time_structure = time.time()
            output_directory = os.path.join(data_compressed_folder, structure_name)
            tools.create_folder(output_directory)
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
                        logging.error(f"Error downloading {url}: {e}")

            elapsed_time_structure = time.time() - start_time_structure
            print(f'  Downloaded all files for {structure_name} ({elapsed_time_structure:.2f} s)')

    def decompress_data(self):
        """Decompresses data files from the compressed folder to the raw data folder."""
        speed_limit = config.SPEED_LIMIT
        source_folder_path = os.path.join(self.data_folder_path, "data_compressed")
        destination_folder_path = os.path.join(self.data_folder_path, "data_raw")

        if not os.path.exists(source_folder_path):
            print(f'Error: The source directory "{source_folder_path}" does not exist.')
            exit()
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        for folder, _, files in os.walk(source_folder_path):
            print(f'    - Decompressing data for {folder}...')
            start_time = time.time()
            for compress_file in files:
                if compress_file.endswith('.xz'):
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
                        logging.error(f"Error decompressing {in_path}: {e}")
                        print(f'· Error in {in_path}: {str(e)}') 

            elapsed_time = time.time() - start_time
            print(f'     Decompressed all files for {folder} ({elapsed_time:.2f} s)')

    def data_download_workflow(self): 
        """Manages the data download workflow, handling existing folders and user choices."""
        

        folder_combinations = {
            (True, True, True): self.handle_all_folders_exist,
            (True, True, False): self.handle_names_compressed_exist,
            (True, False, True): self.handle_names_raw_exist,
            (False, True, True): self.handle_compressed_raw_exist,
            (True, False, False): self.handle_names_exist,
            (False, True, False): self.handle_compressed_exist,
            (False, False, True): self.handle_raw_exist,
            (False, False, False): self.whole_download_workflow,
        }

        folders_exist = (
            os.path.exists(os.path.join(self.data_folder_path, "materials_names")),
            os.path.exists(os.path.join(self.data_folder_path, "data_compressed")),
            os.path.exists(os.path.join(self.data_folder_path, "data_raw")),
        )

        folder_combinations[folders_exist]()

    def handle_all_folders_exist(self):
        """Handles the case where all required folders already exist."""
        confirm = input("WARNING: All required folders already exist. Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
        if confirm.lower() != 'y':
            confirm = input("  · Do you want to use existing data? (y/n): ")
            if confirm.lower() != 'y':
                tools.log_main("  · Data download cancelled. Exiting the program...", save_path=self.run_results_path)
                sys.exit()
            else:
                tools.log_main(f"  · Using existing data in {self.data_folder_path}", save_path=self.run_results_path)
                return
        else:
            self.whole_download_workflow()

    def handle_names_compressed_exist(self):
        """Handles the case where materials_names and data_compressed folders exist."""
        confirm = input("  · Materials names and data compressed folder exists. Do you want to decompress and use it? (y/n): ")
        if confirm.lower() == 'y':
            tools.log_main("  · Decompressing existing data...", save_path=self.run_results_path)
            self.decompress_data()
        else:
            confirm = input("  · Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
            if confirm.lower() != 'y':
                tools.log_main("  · Data download cancelled. Exiting the program...", save_path=self.run_results_path)
                sys.exit()
            else:
                self.whole_download_workflow()

    def handle_names_raw_exist(self):
        """Handles the case where materials_names and data_raw folders exist."""
        confirm = input("  · Materials names and data raw folder exists. Downloading new data may overwrite existing files. Do you want to continue downloading the data? (y/n): ")
        if confirm.lower() != 'y':
            confirm = input("  · Do you want to use existing data? (y/n): ")
            if confirm.lower() != 'y':
                tools.log_main("  · Data download cancelled. Exiting the program...", save_path=self.run_results_path)
                sys.exit()
            else:
                tools.log_main(f"  · Using existing data in {self.data_folder_path}", save_path=self.run_results_path)
                return
        else:
            self.whole_download_workflow()

    def handle_compressed_raw_exist(self):
        """Handles the case where data_compressed and data_raw folders exist."""
        confirm = input("  · Data compressed and data raw folder exists. Downloading new data may overwrite existing files. What do you want to do? (continue/use compressed data/use raw data): ")
        if confirm.lower() == 'use compressed data':
            tools.log_main("  · Decompressing existing data...", save_path=self.run_results_path)
            self.decompress_data()
        elif confirm.lower() == 'use raw data':
            tools.log_main(f"  · Using existing data in {self.data_folder_path}", save_path=self.run_results_path)
            return
        elif confirm.lower() == 'continue':
            self.whole_download_workflow()
        else:
            tools.log_main("  · Invalid entry. Exiting the program...", save_path=self.run_results_path)
            sys.exit()

    def handle_names_exist(self):
        """Handles the case where only the materials_names folder exists."""
        confirm = input("  · Materials names folder already exists. Do you want to update the materials list or keep the existing one? (update/keep): ")
        tools.log_main("  · Starting download. Please, be patient.", save_path=self.run_results_path)
        if confirm.lower() == 'update':
            self.get_names()
        self.download_compressed_data()
        self.decompress_data()
        return

    def handle_compressed_exist(self):
        """Handles the case where only the data_compressed folder exists."""
        confirm = input("  · Data compressed folder exists. Do you want to decompress and use it? (y/n): ")
        if confirm.lower() == 'y':
            tools.log_main("  · Decompressing existing data...", save_path=self.run_results_path)
            self.decompress_data()
        else:
            confirm = input("  · Continuing will overwrite existing data. Do you want to continue downloading the data? (y/n): ")
            if confirm.lower() != 'y':
                tools.log_main("  · Data download cancelled. Exiting the program...", save_path=self.run_results_path)
                sys.exit()
            else:
                self.whole_download_workflow()

    def handle_raw_exist(self):
        """Handles the case where only the data_raw folder exists."""
        confirm = input("  · Data raw folder exists. Downloading new data may overwrite existing files. Do you want to continue downloading the data? (y/n): ")
        if confirm.lower() != 'y':
            confirm = input("  · Do you want to use existing data? (y/n): ")
            if confirm.lower() != 'y':
                tools.log_main("  · Data download cancelled. Exiting the program...", save_path=self.run_results_path)
                sys.exit()
            else:
                tools.log_main(f"  · Using existing data in {self.data_folder_path}", save_path=self.run_results_path)
                return
        else:
            self.whole_download_workflow()

    def whole_download_workflow(self):
        """Performs the complete workflow to download data."""
        confirm = input('WARNING: You are about to start downloading files, this can take up to several hours. Are you sure you want to continue? (y/n): ')
        if confirm.lower() == 'y':
            tools.log_main("  · Starting download. Please, be patient.", save_path=self.run_results_path)
            self.get_names()
            self.download_compressed_data()
            self.decompress_data()
            return
        elif confirm.lower() == 'n':
            tools.log_main("  · File download cancelled. Exiting the program...", save_path=self.run_results_path)
            sys.exit()
        else:
            tools.log_main("  · Invalid entry. Exiting the program...", save_path=self.run_results_path)
            sys.exit() 