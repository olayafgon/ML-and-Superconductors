import sys
import time

from data_handling import data_raw_download, data_raw_read, data_processing
from data_analysis import analysis_plotter, reporter, dos_plotter
from utils import tools

sys.path.append('./../')
import config

_DATA_DOWNLOAD = config.DATA_DOWNLOAD
_READ_DATA_RAW = config.READ_DATA_RAW

def main():

    #START
    start_ = time.time()
    
    # Create results folder
    run_results_path = tools.create_run_results_folder()

    # Download or check data
    if _DATA_DOWNLOAD:
        data_dowloading = data_raw_download.DataDownload(run_results_path)
        data_dowloading.data_download_workflow()

    # Read and save to csv raw_data
    if _READ_DATA_RAW:
        raw_data_reading = data_raw_read.MaterialRawDataRead(run_results_path)
        raw_data_reading.data_raw_read_workflow()
    
    # Procces data
    MaterialsProcessor = data_processing.DataProcessor(run_results_path)
    materials_data, supercon_data = MaterialsProcessor.processor()

    #EDA
    stats_reporter = reporter.StatsReporter(materials_data, run_results_path)
    stats_reporter.stats_report()
    plotter = analysis_plotter.AnalysisPlotter(materials_data, run_results_path)
    plotter.workflow()

    Dos_Plotter = dos_plotter.DosPlotter(materials_data, run_results_path)
    Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=True)
    for bravais in config.STRUCTURES:
        Dos_Plotter.plot_dos(by='Bravais', method='average', is_supercon=None, filter_bravais=[bravais]) 
    Dos_Plotter.plot_dos(by='Bravais', method='median', is_supercon=False)
    for bravais in config.STRUCTURES:
        Dos_Plotter.plot_dos(by='Bravais', method='median', is_supercon=None, filter_bravais=[bravais]) 
    Dos_Plotter.plot_dos(by='ICSD', ICSD=189400)
    Dos_Plotter.plot_dos(by='ICSD', ICSD=608582)
    Dos_Plotter.plot_dos(by='ICSD', ICSD=609426)

    #END
    end_ = time.time()
    tools.log_main(f'Total runtime {(end_-start_):.2f}s ({(end_-start_)/60:.0f}min)', save_path=run_results_path)

if __name__ == '__main__':
    main()