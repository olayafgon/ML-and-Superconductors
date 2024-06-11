import sys
import pandas as pd

sys.path.append('./../')
import config
from utils import tools

class DataProcessor:
    def __init__(self, run_results_path):
        self.dos_csv_path = config.DOS_CSV_PATH
        self.supercon_csv_path = config.SUPERCON_CSV_PATH
        self.run_results_path = run_results_path
        tools.log_main('· MODULE: DataProcessor...', save_path=self.run_results_path)

    def read_supercon_database(self):
        self.supercon_data = pd.read_csv(self.supercon_csv_path, skiprows=1, low_memory=False)
        self.supercon_data.columns = ['chemical_formula', 'critical_temperature_k', 'ICSD', 'synth_doped']
        self.supercon_data['ICSD'] = self.supercon_data['ICSD'].str.replace('ICSD-', '')

    def read_materials_csv(self):
        self.materials_data = pd.read_csv(self.dos_csv_path, low_memory=False)
        self.ICSD_preprocessor()
        self.identify_superconductors()

    def ICSD_preprocessor(self):
        self.materials_data['ICSD'] = pd.to_numeric(self.materials_data['ICSD'], errors='coerce').astype('Int64')
        num_nulos = self.materials_data['ICSD'].isnull().sum()
        len_df_pre = len(self.materials_data)
        print(f"  {num_nulos} rows ({num_nulos/len_df_pre*100:.4f} %) where dropped because of null on ICSD.")
        self.materials_data.dropna(subset=['ICSD'], inplace=True)

    def identify_superconductors(self):
        ICSD_supercon = self.supercon_data.ICSD.unique().astype(int).tolist()
        self.materials_data.insert(5, 'is_superconductor', self.materials_data.ICSD.isin(ICSD_supercon))

    def processor(self):
        self.read_supercon_database()
        self.read_materials_csv()
        return self.materials_data, self.supercon_data

    