import sys

sys.path.append('./../')
import config
from data_analysis import analysis_plotter, reporter, dos_plotter

class EDAPipeline:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        self._DOS_PLOTS = config.DOS_PLOTS
        self._ICSD_PLOTS = config.ICSD_PLOTS

    def perform_eda(self):
        """
        Performs Exploratory Data Analysis (EDA).

        Args:
            materials_data: Material data.
            run_results_path: Path where results will be stored.
        """
        stats_reporter = reporter.StatsReporter(self.materials_data, self.run_results_path)
        stats_reporter.stats_report()
        plotter = analysis_plotter.AnalysisPlotter(self.materials_data, self.run_results_path)
        plotter.workflow()

    def plot_dos(self):
        """
        Generates Density of States (DOS) plots.

        Args:
            materials_data: Material data.
            run_results_path: Path where results will be stored.
            _DOS_PLOTS: Flag to enable/disable DOS plot generation.
            _ICSD_PLOTS: List of ICSD codes for specific DOS plots.
        """
        Dos_Plotter = dos_plotter.DosPlotter(self.materials_data, self.run_results_path)
        Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=True)
        Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=False)
        Dos_Plotter.plot_dos(by='Bravais', method='median', is_supercon=True)
        Dos_Plotter.plot_dos(by='Bravais', method='median', is_supercon=False)
        for bravais in config.STRUCTURES:
            Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=None, filter_bravais=[bravais]) 
            Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=True)
        if self._ICSD_PLOTS != None:
            for ICSD in self._ICSD_PLOTS:
                Dos_Plotter.plot_dos(by='ICSD', ICSD=ICSD)

    def eda_workflow(self):
        self.perform_eda()
        if self._ICSD_PLOTS:
            self.plot_dos()