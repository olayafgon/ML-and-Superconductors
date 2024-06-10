import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import seaborn as sns
import re

pd.set_option('display.max_columns', None)

sys.path.append('./../')
from utils import tools

class Plotter:

    def __init__(self, materials_data, run_results_path):
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman'] 
        mpl.rcParams['text.usetex'] = False
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        tools.log_main('· MODULE: Plotter...', save_path=self.run_results_path)
        print(f'  Graphs and report are being saved in: {run_results_path}')

    def _calculate_stats(self, column_name):
        """Calculates and returns statistics for a given column."""
        value_counts = self.materials_data[column_name].value_counts()
        total_count = len(self.materials_data[column_name])
        true_count = value_counts.get(True, 0)
        true_percentage = (true_count / total_count) * 100
        return true_percentage, total_count, true_count

    def superconductors_stats(self):
        """Calculates and writes superconductor statistics to a report file."""
        true_percentage, total_count, true_count = self._calculate_stats('is_superconductor')
        with open(os.path.join(self.run_results_path, 'stats_report.txt'), 'a') as f:
            f.write(f"Percentage of superconductors: {true_percentage:.2f}%\n")
            f.write(f'Total materials: {total_count}\n')
            f.write(f'Superconducting materials: {true_count}\n')

    def magnetic_stats(self):
        """Calculates and writes magnetic statistics to a report file."""
        true_percentage, total_count, true_count = self._calculate_stats('is_magnetic')
        with open(os.path.join(self.run_results_path, 'stats_report.txt'), 'a') as f:
            f.write(f"Percentage of magnetic: {true_percentage:.2f}%\n")
            f.write(f'Total materials: {total_count}\n')
            f.write(f'Magnetic materials: {true_count}\n')

    def element_stats(self):
        """Calcula y escribe las estadísticas de elementos en el reporte."""
        df = self.materials_data[['material_name', 'is_superconductor']].copy()
        df = self._extract_elements_from_dataframe(df)
        element_df = self._create_element_dataframe(df)

        total_elements = len(element_df['element'].unique())

        superconductor_elements = element_df[element_df['is_superconductor'] == True]['element'].unique()
        num_superconductor_elements = len(superconductor_elements)
        percentage_superconductor_elements = (num_superconductor_elements / total_elements) * 100

        nonsuperconductor_elements = element_df[element_df['is_superconductor'] == False]['element'].unique()
        num_nonsuperconductor_elements = len(nonsuperconductor_elements)
        percentage_nonsuperconductor_elements = (num_nonsuperconductor_elements / total_elements) * 100

        with open(os.path.join(self.run_results_path, 'stats_report.txt'), 'a') as f:
            f.write(f"\n--- Estadísticas de Elementos ---\n")
            f.write(f"Número total de elementos únicos: {total_elements}\n")
            f.write(f"Número de elementos en superconductores: {num_superconductor_elements} ({percentage_superconductor_elements:.2f}%)\n")
            f.write(f"Número de elementos en NO superconductores: {num_nonsuperconductor_elements} ({percentage_nonsuperconductor_elements:.2f}%)\n")

    def stats_report(self):
        """Escribe el reporte completo de estadísticas en un archivo."""
        report_file = os.path.join(self.run_results_path, 'stats_report.txt')
        if not os.path.exists(report_file):
            with open(report_file, 'w') as f:
                f.write(f'···················· STATISTICS ····················\n')
        self.superconductors_stats()
        with open(report_file, 'a') as f:
            f.write(f'····················································\n')
        self.magnetic_stats()
        with open(report_file, 'a') as f:
            f.write(f'····················································\n')
        self.element_stats()
        
    def magnetic_properties_plot(self):
        grouped_data = self.materials_data.groupby(['is_superconductor', 'is_magnetic']).size()
        ax = grouped_data.unstack().plot(kind='bar', stacked=True, colormap='Set2', figsize=(10,4))
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy() 

            percentage = height / grouped_data.sum() * 100
            if height < 0.05 * grouped_data.sum(): 
                ax.text(x + width/2, y + height + 0.01, f'{percentage:.1f}%',  
                        ha='center', va='bottom', fontsize=10, color='black',
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
            else:
                ax.text(x + width/2, y + height/2, f'{percentage:.1f}%', 
                        ha='center', va='center', fontsize=10, color='black',
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        plt.title('Propiedades Magnéticas')
        plt.xlabel('Material')
        plt.ylabel('Número de muestras')
        plt.xticks([0, 1], ['Superconductor', 'No Superconductor'], rotation=0)
        plt.legend(title='Propiedades Magnéticas', labels=['No Magnético', 'Magnético'])
        plt.tight_layout()
        tools.save_plot(self.run_results_path, 'magnetic_properties')

    def supercon_properties_by_bravais_plot(self):
        grouped_data = self.materials_data.groupby(['bravais_lattice', 'is_superconductor']).size()
        unstacked_data = grouped_data.unstack(fill_value=0)

        ax = unstacked_data.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 5))
        total_samples = len(self.materials_data)

        for i, patch in enumerate(ax.patches[:len(unstacked_data)]): 
            height = patch.get_height() + ax.patches[i + len(unstacked_data)].get_height() 
            percentage = 100 * height / total_samples
            ax.annotate(f'{percentage:.1f}%', 
                        xy=(patch.get_x() + patch.get_width() / 2, height),  
                        ha='center', va='bottom', fontsize=9, color='black')

        plt.title('Redes de Bravais y propiedades Superconductoras')
        plt.xlabel('Red de Bravais')
        plt.ylabel('Número de muestras')
        plt.legend(title='Propiedades Superconductoras', labels=['Superconductor', 'No Superconductor'])
        plt.tight_layout()
        tools.save_plot(self.run_results_path, 'superconducting_properties_by_bravais_lattice')
    
    def _extract_elements_from_dataframe(self, df):
        """Extrae los elementos de la columna 'material_name' del DataFrame."""
        def extract_elements(formula):
            return re.findall(r'[A-Z][a-z]?', formula)
        df['elements'] = df['material_name'].apply(extract_elements)
        return df

    def _create_element_dataframe(self, df):
        """Crea un DataFrame con la información de elementos y superconductividad."""
        element_superconductor = []
        for i, row in df.iterrows():
            for element in row['elements']:
                element_superconductor.append([element, row['is_superconductor']])
        return pd.DataFrame(element_superconductor, columns=['element', 'is_superconductor'])

    def _calculate_element_statistics(self, element_df):
        """Calcula las estadísticas de los elementos."""
        superconductors_count = element_df[element_df['is_superconductor'] == True]['element'].value_counts()
        total_count = element_df['element'].value_counts()
        proportion_superconductors = (superconductors_count / total_count).fillna(0)
        return superconductors_count, proportion_superconductors

    def _plot_element_proportion(self, proportion_data, title, filename):
        """Grafica la proporción de elementos."""
        plt.figure(figsize=(5, 10))
        ax = sns.barplot(x=proportion_data.values, y=proportion_data.index, palette='coolwarm')
        plt.xlabel('Número de elementos presentes / Total de elementos')
        plt.ylabel('Elemento')
        plt.title(title)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, filename)

    def _plot_element_counts(self, counts_data, title, filename):
        """Grafica el conteo de elementos."""
        plt.figure(figsize=(5, 10))
        ax = sns.barplot(x=counts_data.values, y=counts_data.index, palette='coolwarm')
        plt.xlabel('Número de veces que aparece el elemento')
        plt.ylabel('Elemento')
        plt.title(title)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        tools.save_plot(self.run_results_path, filename)

    def _plot_top_elements_overall(self, element_df, top_n=25):
        """Grafica los elementos más comunes en general."""
        total_element_counts = element_df['element'].value_counts()
        top_elements = total_element_counts.nlargest(top_n)

        plt.figure(figsize=(12, 5))
        ax = sns.barplot(x=top_elements.index, y=top_elements.values, palette='coolwarm')
        plt.ylabel('Número de veces que aparece el elemento')
        plt.xlabel('Elemento')
        plt.title(f'Elementos más comunes en el dataset (Top {top_n})')

        total_elementos = len(element_df)
        y_max = ax.get_ylim()[1]
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 200, 
                    f'{int(height)} \n({(height/total_elementos)*100:.1f}%)',
                    ha="center")
        plt.ylim(0, y_max*1.05)

        plt.tight_layout()
        tools.save_plot(self.run_results_path, 'element_analysis_3')

    def element_analysis_plots(self):
        """Genera y guarda las gráficas de análisis de elementos."""
        df = self.materials_data[['bravais_lattice', 'material_name', 
                                  'ICSD', 'fermi_energy', 'is_magnetic', 
                                  'is_superconductor']].copy()
        df = self._extract_elements_from_dataframe(df)
        element_df = self._create_element_dataframe(df)
        superconductors_count, proportion_superconductors = self._calculate_element_statistics(element_df)

        top_proportion_superconductors = proportion_superconductors.nlargest(50).sort_values(ascending=False)
        top_superconductors_counts = superconductors_count.nlargest(50).sort_values(ascending=False)

        self._plot_element_proportion(top_proportion_superconductors, 
                                      'Proporción de elementos\nen materiales superconductores (Top 50)',
                                      'element_analysis_1')
        self._plot_element_counts(top_superconductors_counts, 
                                   'Conteo de elementos en materiales\nsuperconductores (Top 50)', 
                                   'element_analysis_2')
        self._plot_top_elements_overall(element_df)
    
    def workflow(self):
        self.stats_report()
        self.magnetic_properties_plot()
        self.supercon_properties_by_bravais_plot()
        self.element_analysis_plots()