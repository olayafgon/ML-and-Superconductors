import sys
import pandas as pd

pd.set_option('display.max_columns', 100)

sys.path.append('./../')
import config
from utils import tools
from models.autogluon import autogluon_models
from models.hiperparameter_search import hiperameter_exploration
from models.data_model_processing import data_model_processor

class ModelPipeline:
    def __init__(self, materials_data, run_results_path):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        self.model_exploration_options = config.MODEL_EXPLORATION

    def perform_model_exploration(self):
        if self.model_exploration_options != None:
            tools.log_main('· Starting model and hiperparameter exploration with Autogluon...', save_path=self.run_results_path)
            for method in self.model_exploration_options:
                Data_Processor = data_model_processor.DataPreprocessor(self.materials_data)
                if method != 'Hiperparameters_Exploration':
                    Autogluon_Training = autogluon_models.AutogluonTraining(self.run_results_path, method, Data_Processor)
                    Autogluon_Training.autogluon_training_run()
                elif method == 'Hiperparameters_Exploration':
                    Hiperameter_Exploration = hiperameter_exploration.HiperparameterExploration(self.run_results_path, method, Data_Processor)
                    Hiperameter_Exploration.hiperparameter_exploration_run()
        else:
            tools.log_main('· Skipping model exploration...', save_path=self.run_results_path)

    def model_workflow(self):
        self.perform_model_exploration()
