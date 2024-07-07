import sys
import pandas as pd

pd.set_option('display.max_columns', 100)

sys.path.append('./../')
import config
from utils import tools
from models.autogluon import autogluon_models

class ModelPipeline:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        self.model_exploration_options = config.MODEL_EXPLORATION

    def perform_model_exploration(self):
        if self.model_exploration_options != None:
            tools.log_main('· Starting model exploration...', save_path=self.run_results_path)
            for method in self.model_exploration_options:
                Autogluon_Training = autogluon_models.AutogluonTraining(self.materials_data, self.run_results_path, method)
                Autogluon_Training.autogluon_training_workflow()
        else:
            tools.log_main('· Skipping model exploration...', save_path=self.run_results_path)

    def model_workflow(self):
        self.perform_model_exploration()
