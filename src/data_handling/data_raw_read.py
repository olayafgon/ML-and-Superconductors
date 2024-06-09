import os
import sys
import json
import pandas as pd
import numpy as np
import time

from utils import tools
sys.path.append('./../')
import config

class MaterialRawDataRead:
    """Class to process material data files and extract DOS information."""

    def __init__(self, run_results_path):
        tools.log_main(f'MODULE: data_read - Reading data_raw...', save_path=self.run_results_path)
        self.run_results_path = run_results_path
        self.data_folder_path = config.DATA_FOLDER_PATH
        self.data_raw_path = os.path.join(self.data_folder_path, r'data_raw')
        self.dos_csv_path = os.path.join(self.data_folder_path, r'dos_data.csv')
        self.data_dict = {}

    @staticmethod
    def extract_material_data(file_path):
        """Reads material data from a JSON file."""
        with open(file_path, 'r') as f:
            material_data = json.load(f)
        return material_data

    @staticmethod
    def get_material_info(material_data, bravais_lattice):
        """Extracts basic material information."""
        material_list = [bravais_lattice]
        name_parts = material_data['name'].split('_')
        material_list.extend([name_parts[0], name_parts[-1], material_data['Efermi']])
        return material_list

    @staticmethod
    def extract_dos_data(material_data):
        """Extracts DOS data from the material dictionary."""
        energy_respect_fermi = np.array(material_data['tDOS_data']['energy'])
        is_magnetic = 'tDOS' not in material_data['tDOS_data']
        if is_magnetic:
            DOS_grid = (np.array(material_data['tDOS_data']['spin_majority']) - 
                        np.array(material_data['tDOS_data']['spin_minority']))[
                (energy_respect_fermi >= -config.EFERMI_LIMIT) & (
                    energy_respect_fermi <= config.EFERMI_LIMIT
                )
            ][:config.EFERMI_GRID_POINTS]
        else:
            DOS_grid = np.array(material_data['tDOS_data']['tDOS'])[
                (energy_respect_fermi >= -config.EFERMI_LIMIT) & (
                    energy_respect_fermi <= config.EFERMI_LIMIT
                )
            ][:config.EFERMI_GRID_POINTS]
        return is_magnetic, DOS_grid

    def process_material(self, file_path):
        """Processes a single material file and extracts relevant DOS information."""
        bravais_lattice = os.path.basename(os.path.dirname(file_path))
        material_data = self.extract_material_data(file_path)
        material_list = self.get_material_info(material_data, bravais_lattice)
        is_magnetic, DOS_grid = self.extract_dos_data(material_data)
        material_list.extend([is_magnetic, DOS_grid])
        self.data_dict[bravais_lattice].append(material_list)

    def process_all_materials(self):
        """Processes all material data files in the specified directory."""
        for bravais_lattice in os.listdir(self.data_raw_path):
            start = time.time()
            print(f'· Reading data files of: {bravais_lattice}...')

            folder_path = os.path.join(self.data_raw_path, bravais_lattice)
            self.data_dict[bravais_lattice] = []
            for file in os.listdir(folder_path):
                self.process_material(os.path.join(folder_path, file))

            elapsed_time = time.time() - start
            print(f'  {bravais_lattice} completed. Time: {elapsed_time:.2f} s')

    @staticmethod
    def create_dos_df(bravais_data):
        """Creates a DataFrame for DOS data of a single bravais lattice."""
        temp_df = pd.DataFrame(
            bravais_data,
            columns=[
                'bravais_lattice',
                'material_name',
                'ICSD',
                'fermi_energy',
                'is_magnetic',
                'dos_grid',
            ],
        )
        df_dos = temp_df['dos_grid'].apply(pd.Series)

        energy_values = np.linspace(-config.EFERMI_LIMIT, config.EFERMI_LIMIT, config.EFERMI_GRID_POINTS)

        def format_energy(energy):
            if energy == 0:
                return 'DOS_0'
            sign = 'm' if energy < 0 else 'p'
            return f'DOS_{sign}{abs(energy):.2f}'.replace('.', '_')

        df_dos.columns = [format_energy(energy) for energy in energy_values]

        return pd.concat([temp_df.drop('dos_grid', axis=1), df_dos], axis=1)

    def convert_to_df(self):
        """Converts the processed data dictionary into a Pandas DataFrame."""
        all_data_df = pd.DataFrame()
        for _, bravais_data in self.data_dict.items():
            df_formatted = self.create_dos_df(bravais_data)
            all_data_df = pd.concat([all_data_df, df_formatted], ignore_index=True)
        return all_data_df

    def data_raw_read_workflow(self):
        """Processes material data files, extracts DOS information, and creates a DataFrame."""
        self.process_all_materials()
        all_data_df = self.convert_to_df()
        all_data_df.fillna(0.0, inplace=True)
        all_data_df.to_csv(self.dos_csv_path, index=False)
        tools.log_main(f'  · Data read and saved in {self.dos_csv_path}', save_path=self.run_results_path)


        