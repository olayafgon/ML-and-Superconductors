import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.append('./../../')
import config
from utils import tools

class ModelTraining:
    def __init__(self, run_results_path, Data_Processor):
        self.run_results_path = run_results_path
        self.Data_Processor = Data_Processor
        self.target_column = config.TARGET_COLUMN
        self.eval_metric = config.EVAL_METRIC
        self.model_algorithm = config.FINAL_MODEL
        if config.FINAL_MODEL_HIPERPARAMETERS == 'Default':
            self.model_hiperparameters = config.FINAL_MODEL_DEFAULT_HIPERPARAMETERS
        elif config.FINAL_MODEL_HIPERPARAMETERS == 'Custom':
            self.model_hiperparameters = config.FINAL_MODEL_CUSTOM_HIPERPARAMETERS
        self.model_pca = config.FINAL_MODEL_PCA_NUMBER
        self.model_resampling = config.FINAL_MODEL_RESAMPLER
        
    @staticmethod
    def split_resample_data(X, y, resampling_technique=None, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if resampling_technique == 'RandomOverSampler':
            sampling_method = RandomOverSampler()
            X_train, y_train = sampling_method.fit_resample(X_train, y_train)
        elif resampling_technique == 'RandomUnderSampler':
            sampling_method = RandomUnderSampler()
            X_train, y_train = sampling_method.fit_resample(X_train, y_train)
        elif resampling_technique == 'SMOTE':
            sampling_method = SMOTE()
            X_train, y_train = sampling_method.fit_resample(X_train, y_train)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2)
        return X_train, X_test, X_eval, y_train, y_test, y_eval
    
    def train_model(self):
        if self.model_algorithm == 'XGBClassifier':
            early_stopping_rounds = self.model_hiperparameters.pop('early_stopping_rounds', 50)
            eval_metric = self.model_hiperparameters.pop('eval_metric', 'error')
            model = XGBClassifier(**self.model_hiperparameters)
            model.set_params(eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_eval, self.y_eval), (self.X_test, self.y_test)],
                verbose=False
            )
            return model
        elif self.model_algorithm == 'LGBMClassifier':
            model = LGBMClassifier(**self.model_hiperparameters)
            model.fit(
                self.X_train, self.y_train
            )
            return model
        else:
            raise ValueError("FINAL_MODEL must be 'LGBMClassifier' or 'XGBClassifier'")
    
    def model_training_run(self):
        tools.log_main(f'  - Processing data: PCA={self.model_pca} and Resampling={self.model_resampling}...', save_path=self.run_results_path)
        X, y = self.Data_Processor.preprocess_data()
        X, pca_columns = self.Data_Processor.apply_pca(self.model_pca, X)
        X, y = self.Data_Processor.basic_processing(X, y, pca_columns)
        self.X_train, self.X_test, self.X_eval, self.y_train, self.y_test, self.y_eval = self.split_resample_data(X, y, resampling_technique=self.model_resampling, test_size=0.2, random_state=42)
        tools.log_main(f'  - Training model: {self.model_algorithm}...', save_path=self.run_results_path)
        tools.log_main(f'    Hiperparameters: {self.model_hiperparameters}...', save_path=self.run_results_path)
        start_ = time.time()
        model = self.train_model()
        end_ = time.time()
        tools.log_main(f'  - Training time: {(end_-start_):.2f}s ({(end_-start_)/60:.0f}min)', save_path=self.run_results_path)
        return model, self.X_test, self.y_test
        