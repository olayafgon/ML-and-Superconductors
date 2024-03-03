import os
import sys
import shutil
from datetime import datetime
from utils import tools

sys.path.append('./../')
import config


class Main_processes():

    def __init__(self):
        self.all_results_path = config.RESULTS_FOLDER
        self.run_results_path = os.path.join(self.all_results_path, str(datetime.now().strftime('%Y%m%d_%H_%M')))

    def create_run_results_folder(self):
        '''
        Creates the results folder for the run and return its path.

        RETURNS:
            - str: Path to the created results folder.
        '''
        tools.create_folder(self.all_results_path)
        tools.create_folder(self.run_results_path)
        shutil.copy('../config.py', self.run_results_path)
        tools.log_main(f'MODULE: Results folder created: {self.run_results_path}', save_path=self.run_results_path)
        return self.run_results_path
