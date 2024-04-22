import os
import sys
import json
import pandas as pd
import numpy as np
import time

from utils import tools
sys.path.append('./../')
import config

class MaterialDataProcessor:
    """Class to process material data files and extract DOS information."""

    def __init__(self, run_results_path):
        self.run_results_path = run_results_path
        self.data_folder_path = config.DATA_FOLDER_PATH
        self.data_raw_path = os.path.join(self.data_folder_path, r'data_raw')
        self.dos_csv_path = os.path.join(self.data_folder_path, r'dos_data.csv')
        self.efermi_limit = config.EFERMI_LIMIT
        self.data_dict = {}
        tools.log_main(f'MODULE: data_read - Reading data_raw...', save_path=self.run_results_path)

    def process_material(self, file_path):
        """Processes a single material file and extracts relevant DOS information.

        Args:
            file_path (str): Path to the material data file.
        """
        with open(file_path, 'r') as f:
            material_data = json.load(f)

        material_list = []
        bravais_lattice = os.path.basename(os.path.dirname(file_path))
        material_list.append(bravais_lattice)

        name_parts = material_data['name'].split('_')
        material_list.append(name_parts[0])  # Name
        material_list.append(name_parts[-1])  # ICSD
        material_list.append(material_data['Efermi'])

        energy_respect_fermi = np.array(material_data['tDOS_data']['energy'])

        if 'tDOS' in material_data['tDOS_data']:
            material_list.append(False)  # No magnetic
            DOS_data = np.array(material_data['tDOS_data']['tDOS'])
            desired_indices = (energy_respect_fermi >= -self.efermi_limit) & (
                energy_respect_fermi <= self.efermi_limit
            )
            DOS_grid = DOS_data[desired_indices][:1999]
        else:
            material_list.append(True)  # Magnetic
            DOS_spin_majority = np.array(material_data['tDOS_data']['spin_majority'])
            DOS_spin_minority = np.array(material_data['tDOS_data']['spin_minority'])
            desired_indices = (energy_respect_fermi >= -self.efermi_limit) & (
                energy_respect_fermi <= self.efermi_limit
            )
            DOS_grid = (DOS_spin_majority - DOS_spin_minority)[desired_indices][:1999]

        material_list.append(DOS_grid)

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

            end = time.time()
            elapsed_time = end - start
            print(f'  {bravais_lattice} completed. Time: {elapsed_time:.2f} s')

    def convert_to_df(self):
        """Converts the processed data dictionary into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with all the processed material data.
        """
        all_data_df = pd.DataFrame()
        for _, bravais_data in self.data_dict.items():
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
            df_dos.columns = [f'DOS_{i}' for i in range(len(df_dos.columns))]
            df_formatted = pd.concat([temp_df.drop('dos_grid', axis=1), df_dos], axis=1)
            all_data_df = pd.concat([all_data_df, df_formatted], ignore_index=True)
        return all_data_df

    def data_raw_read_workflow(self):
        """Processes material data files, extracts DOS information, and creates a DataFrame."""
        self.process_all_materials()
        all_data_df = self.convert_to_df()
        all_data_df.fillna(0.0, inplace=True)
        all_data_df.to_csv(self.dos_csv_path, index=False)
        tools.log_main(f'  · Data read and saved in {self.dos_csv_path}', save_path=self.run_results_path)


        