import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns
import re

pd.set_option('display.max_columns', None)

from data_analysis import analysis_utils
sys.path.append('./../')
from utils import tools

class StatsReporter:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        tools.log_main('· MODULE: StatsReporter...', save_path=self.run_results_path)

    def _calculate_stats(self, column_name):
        """Calculates and returns statistics for a given column."""
        value_counts = self.materials_data[column_name].value_counts()
        total_count = len(self.materials_data[column_name])
        true_count = value_counts.get(True, 0)
        true_percentage = (true_count / total_count) * 100
        return true_percentage, total_count, true_count

    def superconductors_stats(self, report_file):
        """Calculates and writes superconductor statistics to a report file."""
        true_percentage, total_count, true_count = self._calculate_stats('is_superconductor')
        tools.write_to_report(report_file, f"Percentage of superconductors: {true_percentage:.2f}%\n")
        tools.write_to_report(report_file, f'Total materials: {total_count}\n')
        tools.write_to_report(report_file, f'Superconducting materials: {true_count}\n')

    def magnetic_stats(self, report_file):
        """Calculates and writes magnetic statistics to a report file."""
        true_percentage, total_count, true_count = self._calculate_stats('is_magnetic')
        tools.write_to_report(report_file, f"Percentage of magnetic: {true_percentage:.2f}%\n")
        tools.write_to_report(report_file, f'Total materials: {total_count}\n')
        tools.write_to_report(report_file, f'Magnetic materials: {true_count}\n')

    def element_stats(self, report_file):
        """Calculates and writes element statistics to the report."""
        df = self.materials_data[['material_name', 'is_superconductor']].copy()
        df = analysis_utils.extract_elements_from_dataframe(df)
        element_df = analysis_utils.create_element_dataframe(df)

        total_elements = len(element_df['element'].unique())

        superconductor_elements = element_df[element_df['is_superconductor'] == True]['element'].unique()
        num_superconductor_elements = len(superconductor_elements)
        percentage_superconductor_elements = (num_superconductor_elements / total_elements) * 100

        nonsuperconductor_elements = element_df[element_df['is_superconductor'] == False]['element'].unique()
        num_nonsuperconductor_elements = len(nonsuperconductor_elements)
        percentage_nonsuperconductor_elements = (num_nonsuperconductor_elements / total_elements) * 100

        tools.write_to_report(report_file, f"\n--- Estadísticas de Elementos ---\n")
        tools.write_to_report(report_file, f"Número total de elementos únicos: {total_elements}\n")
        tools.write_to_report(report_file, f"Número de elementos en superconductores: {num_superconductor_elements} ({percentage_superconductor_elements:.2f}%)\n")
        tools.write_to_report(report_file, f"Número de elementos en NO superconductores: {num_nonsuperconductor_elements} ({percentage_nonsuperconductor_elements:.2f}%)\n")

    def stats_report(self):
        """Writes the complete statistics report to a file."""
        report_file = os.path.join(self.run_results_path, 'stats_report.txt')
        if not os.path.exists(report_file):
            tools.write_to_report(report_file, f'···················· STATISTICS ····················\n')
        self.superconductors_stats(report_file)
        tools.write_to_report(report_file, f'····················································\n')
        self.magnetic_stats(report_file)
        tools.write_to_report(report_file, f'····················································\n')
        self.element_stats(report_file)
        tools.log_main(f'  - Report saved: {report_file}', save_path=self.run_results_path)