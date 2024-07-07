import sys
import time

from data_handling import data_raw_download, data_raw_read, data_processing
from pipeline import eda_pipeline, model_pipeline
from utils import tools

sys.path.append('./../')
import config

_DATA_DOWNLOAD = config.DATA_DOWNLOAD
_READ_DATA_RAW = config.READ_DATA_RAW
_RUN_EDA = config.RUN_EDA


def main():

    #START
    start_ = time.time()
    
    # Create results folder
    run_results_path = tools.create_run_results_folder()

    # Download or check data
    if _DATA_DOWNLOAD:
        Data_Dowload = data_raw_download.DataDownload(run_results_path)
        Data_Dowload.data_download_workflow()

    # Read and save to csv raw_data
    if _READ_DATA_RAW:
        Material_Raw_Data_Read = data_raw_read.MaterialRawDataRead(run_results_path)
        Material_Raw_Data_Read.data_raw_read_workflow()
    
    # Procces data
    Data_Processor = data_processing.DataProcessor(run_results_path)
    materials_data, supercon_data = Data_Processor.processor()

    #EDA
    if _RUN_EDA:
        EDA_Pipeline = eda_pipeline.EDAPipeline(materials_data, run_results_path)
        EDA_Pipeline.eda_workflow()

    # Model exploration
    Model_Pipeline = model_pipeline.ModelPipeline(materials_data, run_results_path)
    Model_Pipeline.model_workflow()

    #END
    end_ = time.time()
    tools.log_main(f'Total runtime {(end_-start_):.2f}s ({(end_-start_)/60:.0f}min)', save_path=run_results_path)

if __name__ == '__main__':
    main()