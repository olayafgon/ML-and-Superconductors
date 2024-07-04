import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tabulate import tabulate
from autogluon.tabular import TabularDataset, TabularPredictor
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import time

pd.set_option('display.max_columns', 100)

sys.path.append('./../../')
import config
from utils import tools


class AutogluonTraining:
    def __init__(self, materials_data, run_results_path, method):
        self.materials_data = materials_data
        self.run_results_path = run_results_path
        self.method = method

    def process_data(self):
        self.data = self.materials_data.copy()
        self.data.dropna(inplace=True)
        self.data['is_superconductor'] = self.data['is_superconductor'].astype(bool)
    
    def define_columns(self):
        start_col = f"DOS_m{abs(config.EFERMI_LIMIT):02.0f}_00"  
        end_col = f"DOS_p{abs(config.EFERMI_LIMIT):02.0f}_00"
        self.dos_cols = self.data.loc[:, start_col:end_col].columns.tolist()
        self.categorical_cols = ['bravais_lattice']
        self.numerical_cols = ['fermi_energy', 'is_magnetic']
        self.target_col = ['is_superconductor']

    def basic_processing(self):
        X = self.data.drop(self.target_col, axis=1)
        X = X.fillna(0) 
        y = self.data[self.target_col]
        y = y.iloc[:, 0]
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = RobustScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_cols),
                ('num', numerical_transformer, self.numerical_cols + self.dos_cols)  
            ]
        )  
        X_preprocessed = preprocessor.fit_transform(X)
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())

        return X_preprocessed, y
    
    @staticmethod
    def save_autogluon_results(performance, leaderboard, path):
        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_df.to_csv(os.path.join(path, 'autogluon_raw', 'leaderboard.csv'), index=False)
        with open(os.path.join(path, 'autogluon_raw', 'report.txt'), 'w') as f:
            f.write(f'Performance:\n{performance}')

    def autogluon_run(self, X, y, method_path):
        autogluon_path = os.path.join(method_path, 'Autogluon')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_data = pd.concat([pd.DataFrame(X_train).reset_index(drop=True), pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
        test_data = pd.concat([pd.DataFrame(X_test).reset_index(drop=True), pd.DataFrame(y_test).reset_index(drop=True)], axis=1)

        predictor = TabularPredictor(label='is_superconductor', 
                                    problem_type='binary', 
                                    eval_metric='f1', 
                                    path=autogluon_path
                                    ).fit(
                                        TabularDataset(train_data),
                                        presets='medium_quality'
                                        )
        performance = predictor.evaluate(TabularDataset(test_data))
        leaderboard = predictor.leaderboard(TabularDataset(test_data), extra_metrics=['accuracy', 'roc_auc', 'precision', 'recall'], silent=True)
        self.save_autogluon_results(performance, leaderboard, method_path)


    def autogluon_training_workflow(self):
        # Añadir logs
        self.process_data()
        self.define_columns()
        if self.method == 'Basic_Autogluon':
            X, y = self.basic_processing()
            autogluon_path = os.path.join(self.run_results_path, self.method)
            self.autogluon_run(X, y, autogluon_path)