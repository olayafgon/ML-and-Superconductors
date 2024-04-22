'''

'''

import os
import sys
import time
from datetime import datetime

from data_handling import data_download
from utils import tools


sys.path.append('./../')
import config

_DATA_DOWNLOAD = config.DATA_DOWNLOAD


def main():

    #START
    start_ = time.time()
    
    # Create results folder
    run_results_path = tools.create_run_results_folder()

    # Download or check data
    if _DATA_DOWNLOAD:
        data_dowloading = data_download.data_download(run_results_path)
        data_dowloading.data_download_workflow()

    # Read and transform data to csv




    #END
    end_ = time.time()
    tools.log_main(f'Total runtime {(end_-start_):.2f}s ({(end_-start_)/60:.0f}min)', save_path=run_results_path)

    

if __name__ == '__main__':
    main()