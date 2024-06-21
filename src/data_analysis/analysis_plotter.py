import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from data_analysis import analysis_utils
sys.path.append('./../')
from utils import tools

class AnalysisPlotter:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman'] 
        mpl.rcParams['text.usetex'] = False
        self.plots_path = os.path.join(self.run_results_path, 'Analysis_plots')
        tools.create_folder(self.plots_path, delete_old = True)
        tools.log_main('· MODULE: AnalysisPlotter...', save_path=self.run_results_path)

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
        tools.save_plot(self.plots_path, filename)

    def magnetic_properties_plot(self):
        """Generates and saves magnetic properties plot."""
        grouped_data = self.materials_data.groupby(['is_superconductor', 'is_magnetic']).size()
        self._create_stacked_bar_plot(grouped_data, 'Propiedades superconductoras y magnéticas', 'Propiedad superconductora', 
                                      'Número de materiales', 'Propiedad magnética', 'superconducting_magnetic_properties')

    def supercon_properties_by_bravais_plot(self):
        """Generates and saves superconductor properties by bravais lattice plot."""
        grouped_data = self.materials_data.groupby(['bravais_lattice', 'is_superconductor']).size()
        total_samples = len(self.materials_data)
        ax = grouped_data.unstack(fill_value=0).plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 5))
        tools.barplot_percentages_labels(ax, grouped_data.unstack(fill_value=0), total=total_samples)
        plt.title('Superconductividad y redes de Bravais')
        plt.xlabel('Red de Bravais')
        plt.ylabel('Número de materiales')
        plt.legend(title='Propiedad superconductora', labels=['No superconductor', 'Superconductor'])
        plt.tight_layout()
        tools.save_plot(self.plots_path, 'superconducting_properties_by_bravais_lattice')

    def _calculate_element_statistics(self, element_df):
        """Calculates element statistics."""
        superconductors_count = element_df[element_df['is_superconductor'] == True]['element'].value_counts()
        total_count = element_df['element'].value_counts()
        proportion_superconductors = (superconductors_count / total_count).fillna(0)
        return superconductors_count, proportion_superconductors

    def _plot_element_statistics(self, counts_data, proportion_data, n=30):
        """Plots element statistics."""
        plt.figure(figsize=(5, 8))
        top_count = counts_data.nlargest(n)
        ax = sns.barplot(x=top_count.values, y=top_count.index, palette='coolwarm')
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Nº de superconductores con el elemento')
        plt.ylabel('Elemento químico')
        plt.title(f'Frecuencia de elementos químicos en\nmateriales superconductores (Top {n})')
        plt.tight_layout()
        tools.save_plot(self.plots_path, 'supercon_elements_count')

        plt.figure(figsize=(5, 8))
        top_proportion = proportion_data.nlargest(n)
        ax = sns.barplot(x=top_proportion.values, y=top_proportion.index, palette='coolwarm')
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Nº de superconductores con el elemento / Nº de materiales con el elemento')
        plt.ylabel('Elemento químico')
        plt.title(f'Proporción de superconductores para\ncada elemento químico (Top {n})')
        plt.tight_layout()
        tools.save_plot(self.plots_path, 'supercon_elements_proportion')

    def _plot_top_elements_overall(self, element_df, top_n=25):
        """Plots the most common elements overall."""
        total_element_counts = element_df['element'].value_counts()
        top_elements = total_element_counts.nlargest(top_n)
        total_elements = len(element_df)
        plt.figure(figsize=(12, 5))
        ax = sns.barplot(x=top_elements.index, y=top_elements.values, palette='coolwarm')
        tools.barplot_percentages_labels(ax, top_elements, total=total_elements)
        plt.ylabel('Nº de materiales')
        plt.xlabel('Elemento químico')
        plt.title(f'Elementos químicos más comunes en el conjunto de datos (Top {top_n})')
        plt.ylim(0, (ax.get_ylim()[1])*1.05)
        plt.tight_layout()
        tools.save_plot(self.plots_path, 'general_elements_count')

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

        self._plot_element_statistics(top_superconductors_counts, top_proportion_superconductors)
        self._plot_top_elements_overall(element_df)

    def workflow(self):
        """Generates all plots."""
        self.magnetic_properties_plot()
        self.supercon_properties_by_bravais_plot()
        self.element_analysis_plots()