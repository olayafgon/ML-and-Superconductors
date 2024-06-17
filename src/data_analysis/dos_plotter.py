import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./../')
from utils import tools

class DosPlotter:
    def __init__(self, data_materials, run_results_path):
        self.data_materials = data_materials
        self.run_results_path = run_results_path
        self.plots_path = os.path.join(self.run_results_path, 'DOS_plots')
        tools.create_folder(self.plots_path, delete_old = True)
        tools.log_main('· MODULE: DosPlotter...', save_path=self.run_results_path)

    def _prepare_bravais_data(self, method, is_supercon, filter_bravais=None):
        """Prepares data for plotting DOS by Bravais lattice."""
        if method is None:
            raise ValueError("Method must be specified for 'Bravais' plots ('average' or 'median')")
        if filter_bravais is not None:
            data_plot = self.data_materials[self.data_materials["bravais_lattice"].isin(filter_bravais)].copy()
        else:
            data_plot = self.data_materials.copy()

        if is_supercon is not None:
            data_plot = data_plot[data_plot.is_superconductor == is_supercon]
        grouped_data = data_plot.groupby('bravais_lattice')
        return grouped_data

    def _calculate_dos_by_method(self, group_data, method):
        """Calculates DOS based on the specified method (average or median)."""
        if method == 'average':
            dos_values = group_data.loc[:, 'DOS_m15_00':'DOS_p15_00'].mean(axis=0)
        elif method == 'median':
            dos_values = group_data.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)
        else:
            raise ValueError("Invalid method. Choose 'average' or 'median'.")
        return dos_values

    def _plot_bravais_dos_data(self, grouped_data, method, is_supercon):
        """Plots DOS data for Bravais lattices."""
        plt.figure(figsize=(12, 5))
        for lattice, group_data in grouped_data:
            dos_values_super = self._calculate_dos_by_method(group_data[group_data.is_superconductor == True], method)
            dos_values_nonsuper = self._calculate_dos_by_method(group_data[group_data.is_superconductor == False], method)

            if is_supercon is None:
                plt.plot(dos_values_super, label=f"{lattice} (Superconductor)")
                plt.plot(dos_values_nonsuper, label=f"{lattice} (Non-Superconductor)")
                plt.title(f"{method.capitalize()} DOS for Superconductors and Non-Superconductors by Bravais Lattices")
            elif is_supercon:
                plt.plot(dos_values_super, label=f"{lattice} (Superconductor)")
                plt.title(f"{method.capitalize()} DOS for Superconductors by Bravais Lattices")
            else:
                plt.plot(dos_values_nonsuper, label=f"{lattice} (Non-Superconductor)")
                plt.title(f"{method.capitalize()} DOS for Non-Superconductors by Bravais Lattices")
        self._set_plot_labels_and_axes()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside the plot
        self._set_dynamic_ylim()

    def _set_plot_labels_and_axes(self):
        """Sets common labels and axes for Bravais DOS plots."""
        num_cols = len(self.data_materials.columns)
        center_index = num_cols // 2
        plt.axhline(y=0, color='black', linestyle=':')
        plt.axvline(x=center_index, color='black', linestyle=':')
        plt.xticks([0, center_index, num_cols - 1], labels=['$E_f-15eV$', '0', '$E_f+15eV$'])
        plt.ylabel("Density of States (DOS)")
        plt.xlabel("Energy - Fermi Energy")
        plt.grid(False)

    def _set_dynamic_ylim(self):
        """Sets the y-axis limits dynamically based on the highest value,
           excluding the first and last 50 points."""
        ax = plt.gca()
        lines = ax.lines
        max_y = -np.inf
        for line in lines:
            y_data = line.get_ydata()
            if len(y_data) >= 100:
                max_y = max(max_y, max(y_data[50:-50]))
        if max_y != -np.inf:
            plt.ylim(-2, max_y * 1.1) 

    def _plot_icsd_dos(self, ICSD):
        """Plots DOS for a specific material based on its ICSD number."""
        if ICSD is None:
            raise ValueError("ICSD must be specified for 'ICSD' plots.")

        filtered_data = self.data_materials[self.data_materials['ICSD'] == ICSD]
        if not filtered_data.empty:
            material_data = filtered_data.iloc[0]
            dos_values = material_data.loc['DOS_m15_00':'DOS_p15_00']

            plt.figure(figsize=(12, 4))
            plt.plot(dos_values)
            self._set_icsd_plot_details(material_data, dos_values)
            self._set_dynamic_ylim()
        else:
            print(f"No data found for ICSD: {ICSD}")

    def _set_icsd_plot_details(self, material_data, dos_values):
        """Sets plot details for ICSD DOS plots."""
        print('Material information')
        print(material_data[:6])

        center_index = len(dos_values) // 2
        plt.axvline(x=center_index, color='black', linestyle=':')
        plt.axhline(y=0, color='black', linestyle=':')
        plt.xticks([0, center_index, len(dos_values) - 1], labels=['$E_f-15eV$', '0', '$E_f+15eV$'])
        plt.ylabel("Density of States (DOS)")
        plt.xlabel("Energy - Fermi Energy")
        plt.title("DOS vs Energy ICSD: {}".format(material_data['ICSD']))
        plt.grid(False)
        plt.ylim(-5, 100)

    def plot_dos(self, by='Bravais', method=None, is_supercon=None, filter_bravais=None, ICSD=None):
        """
        Plots DOS vs Energy based on selected criteria:

        Args:
            by (str): 'Bravais' or 'ICSD', indicating the grouping method.
            method (str): 'average' or 'median', used only if by='Bravais'.
            is_supercon (bool): True, False, or None, used only if by='Bravais'.
            filter_bravais (list): List of Bravais lattices to filter, used only if by='Bravais'.
            ICSD (int): ICSD number for a specific material, used only if by='ICSD'.
        """
        if by == 'Bravais':
            grouped_data = self._prepare_bravais_data(method, is_supercon, filter_bravais)
            self._plot_bravais_dos_data(grouped_data, method, is_supercon)
            filename = f"dos_{by}_{method}"
            if is_supercon is not None:
                filename += f"_{'super' if is_supercon else 'nonsuper'}"
            if filter_bravais is not None:
                filename += f"_{'_'.join(filter_bravais)}"
            tools.save_plot(self.run_results_path, filename)

        elif by == 'ICSD':
            self._plot_icsd_dos(ICSD)
            filename = f"dos_{by}_{ICSD}"
            tools.save_plot(self.plots_path, filename)

        else:
            raise ValueError("Invalid 'by' value. Choose 'Bravais' or 'ICSD'.")