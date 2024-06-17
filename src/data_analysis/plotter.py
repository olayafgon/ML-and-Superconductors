import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

pd.set_option('display.max_columns', None)

from data_analysis import analysis_utils
sys.path.append('./../')
from utils import tools

class Plotter:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman'] 
        mpl.rcParams['text.usetex'] = False

    def _create_stacked_bar_plot(self, grouped_data, title, xlabel, ylabel, legend_title, filename):
        """Creates a stacked bar plot with percentages."""
        ax = grouped_data.unstack().plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 4))
        tools.stack_plot_percentages_labels(ax, grouped_data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=0)
        plt.legend(title=legend_title)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, filename)

    def magnetic_properties_plot(self):
        """Generates and saves magnetic properties plot."""
        grouped_data = self.materials_data.groupby(['is_superconductor', 'is_magnetic']).size()
        self._create_stacked_bar_plot(grouped_data, 'Magnetic Properties', 'Material', 'Number of Samples', 'Magnetic Properties', 'magnetic_properties')

    def supercon_properties_by_bravais_plot(self):
        """Generates and saves superconductor properties by bravais lattice plot."""
        grouped_data = self.materials_data.groupby(['bravais_lattice', 'is_superconductor']).size()
        total_samples = len(self.materials_data)
        ax = grouped_data.unstack(fill_value=0).plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 5))
        tools.barplot_percentages_labels(ax, grouped_data.unstack(fill_value=0), total=total_samples)
        plt.title('Bravais Lattices and Superconducting Properties')
        plt.xlabel('Bravais Lattice')
        plt.ylabel('Number of Samples')
        plt.legend(title='Superconducting Properties', labels=['Superconductor', 'Non-Superconductor'])
        plt.tight_layout()
        tools.save_plot(self.run_results_path, 'superconducting_properties_by_bravais_lattice')

    def _calculate_element_statistics(self, element_df):
        """Calculates element statistics."""
        superconductors_count = element_df[element_df['is_superconductor'] == True]['element'].value_counts()
        total_count = element_df['element'].value_counts()
        proportion_superconductors = (superconductors_count / total_count).fillna(0)
        return superconductors_count, proportion_superconductors

    def _plot_element_statistics(self, counts_data, proportion_data, title, filename):
        """Plots element statistics."""
        plt.figure(figsize=(5, 10))
        ax = sns.barplot(x=counts_data.values, y=counts_data.index, palette='coolwarm')
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Number of times element appears')
        plt.ylabel('Element')
        plt.title(title)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, filename)

        plt.figure(figsize=(5, 10))
        ax = sns.barplot(x=proportion_data.values, y=proportion_data.index, palette='coolwarm')
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Number of elements present / Total elements')
        plt.ylabel('Element')
        plt.title(title)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, filename)

    def _plot_top_elements_overall(self, element_df, top_n=25):
        """Plots the most common elements overall."""
        total_element_counts = element_df['element'].value_counts()
        top_elements = total_element_counts.nlargest(top_n)
        total_elements = len(element_df)
        plt.figure(figsize=(12, 5))
        ax = sns.barplot(x=top_elements.index, y=top_elements.values, palette='coolwarm')
        tools.barplot_percentages_labels(ax, top_elements, total=total_elements)
        plt.ylabel('Number of times element appears')
        plt.xlabel('Element')
        plt.title(f'Most Common Elements in Dataset (Top {top_n})')
        plt.ylim(0, (ax.get_ylim()[1])*1.05)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, 'element_analysis_3')

    def element_analysis_plots(self):
        """Generates and saves element analysis plots."""
        df = self.materials_data[['bravais_lattice', 'material_name', 
                                  'ICSD', 'fermi_energy', 'is_magnetic', 
                                  'is_superconductor']].copy()
        df = analysis_utils.extract_elements_from_dataframe(df)
        element_df = analysis_utils.create_element_dataframe(df)
        superconductors_count, proportion_superconductors = self._calculate_element_statistics(element_df)
        
        top_proportion_superconductors = proportion_superconductors.nlargest(50).sort_values(ascending=False)
        top_superconductors_counts = superconductors_count.nlargest(50).sort_values(ascending=False)

        self._plot_element_statistics(top_superconductors_counts, top_proportion_superconductors, 
                                      'Element Statistics in Superconducting Materials (Top 50)',
                                      'element_analysis_1')
        self._plot_top_elements_overall(element_df)

    def workflow(self):
        """Generates all plots."""
        self.magnetic_properties_plot()
        self.supercon_properties_by_bravais_plot()
        self.element_analysis_plots()