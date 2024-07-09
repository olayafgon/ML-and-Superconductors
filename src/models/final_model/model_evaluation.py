import sys

sys.path.append('./../')
import config
from utils import tools


class ModelEvaluation:
    def __init__(self, run_results_path, model, X_test, y_test):
        self.run_results_path = run_results_path
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        
    def model_training_run(self):
        pass
        