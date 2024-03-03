'''
This module define a workflow to download data from Aflowlib (http://aflowlib.duke.edu).

Requires some configurations from config.py:
    - STRUCTURES
    - DATA_FOLDER_PATH
'''

import os
import sys
import subprocess
import requests
import urllib.request
from utils import tools

sys.path.append('./../')
import config

class data_download():

    def __init__(self, run_results_path):
        '''
        Inicializate module.
        '''
        self.run_results_path = run_results_path
        self.structures = config.STRUCTURES
        self.data_folder_path = config.DATA_FOLDER_PATH
        self.data_dowload_check = config.DATA_DOWNLOAD
        self.aflowlib_link = 'http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/'


    def get_names(self):
        '''
        Creates a folder in data folder with files containing the names of all the materials available in aflowlib for each bravais lattice.
        '''
        # Creates folder
        materials_names_folder = os.path.join(self.data_folder_path, 'materials_names')
        tools.create_folder(materials_names_folder)
    
        for structure in self.structures:
            # Download info files from librery
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
        # TODO: Añadir oocion para elegir archivo a descargar
        # TODO: Añadir checjk de errores en la descarga
        
        materials_names_folder = os.path.join(self.data_folder_path, "materials_names")
        data_compressed_folder = os.path.join(self.data_folder_path, "data_compressed")

        for structure_name in self.structures:
            # Directorio de salida
            output_directory = os.path.join(data_compressed_folder, structure_name)
            tools.create_folder(output_directory)

            # Para cada estructura XXX sustituir por:  f=Data_names.2.Modified/XXX.m.dat
            entry_file = os.path.join(materials_names_folder, f'{structure_name}_names.txt')

            # Bucle para la descarga de los archivos
            with open(entry_file, "r") as file:
                for line in file:
                    # Eliminar el carácter de nueva línea de la línea
                    line = line.strip()
                    
                    j = 0
                    for aa in line:
                        if aa == ".":
                            break
                        j += 1
                    
                    aa = structure_name
                    bb = line[0:j]
                    
                    url = f'http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/{aa}/{bb}/DOSCAR.static.xz'
                    file_path = os.path.join(output_directory, f"{aa}_{bb}.xz")
                    
                    # Imprime la URL antes de descargar
                    print(f"Descargando: {url}")

                    # Descarga el archivo desde la URL
                    urllib.request.urlretrieve(url, file_path)




    def decompress_data(self):
        # TODO: 
        pass



    def data_download_workflow(self):
        """
        Manages the data download workflow.

        This function checks the existence of required folders and guides the user through the data download process.
        It prompts the user for input based on the existing folders and performs appropriate actions accordingly.

        Args:
            self: Instance of the class containing necessary attributes.

        Returns:
            None
        """
        # Construct paths to required folders
        materials_names_folder = os.path.join(self.data_folder_path, "materials_names")
        data_compressed_folder = os.path.join(self.data_folder_path, "data_compressed")
        data_raw_folder = os.path.join(self.data_folder_path, "data_raw")

        # Complete workflow to dowload data
        def whole_download_workflow():
            confirm = input('WARNING: You are about to start downloading files, this can take up to several hours. Are you sure you want to continue? (y/n): ')
            if confirm.lower() == 'y':
                print("  · Starting download. Please, be patient.")
                self.get_names()
                self.download_compressed_data()
                self.decompress_data()
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
        if os.path.exists(materials_names_folder) and os.path.exists(data_raw_folder):
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
        if os.path.exists(data_compressed_folder) and os.path.exists(data_raw_folder):
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
