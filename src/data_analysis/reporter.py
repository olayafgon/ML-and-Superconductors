import sys
import pandas as pd
import os

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

    def _write_stats_to_report(self, report_file, column_name, title):
        """Writes statistics for a given column to the report file."""
        true_percentage, total_count, true_count = self._calculate_stats(column_name)
        tools.write_to_report(report_file, f"\n--- {title} ---\n")
        tools.write_to_report(report_file, f"Percentage of {title.lower()}: {true_percentage:.2f}%\n")
        tools.write_to_report(report_file, f'Total materials: {total_count}\n')
        tools.write_to_report(report_file, f'{title.title()} materials: {true_count}\n')

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

        tools.write_to_report(report_file, f"\n--- Element Statistics ---\n")
        tools.write_to_report(report_file, f"Total number of unique elements: {total_elements}\n")
        tools.write_to_report(report_file, f"Number of elements in superconductors: {num_superconductor_elements} ({percentage_superconductor_elements:.2f}%)\n")
        tools.write_to_report(report_file, f"Number of elements in non-superconductors: {num_nonsuperconductor_elements} ({percentage_nonsuperconductor_elements:.2f}%)\n")

    def stats_report(self):
        """Writes the complete statistics report to a file."""
        report_file = os.path.join(self.run_results_path, 'stats_report.txt')
        if not os.path.exists(report_file):
            tools.write_to_report(report_file, f'·········· STATISTICS ··········\n')
        self._write_stats_to_report(report_file, 'is_superconductor', 'Superconductors')
        self._write_stats_to_report(report_file, 'is_magnetic', 'Magnetic Properties')
        self.element_stats(report_file)