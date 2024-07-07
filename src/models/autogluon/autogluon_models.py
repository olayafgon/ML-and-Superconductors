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
        self.target_column = config.TARGET_COLUMN
        self.eval_metric = config.EVAL_METRIC
        self.resampling_techniques = [
            ("none", None),
            ("RandomUnderSampler", RandomUnderSampler()),
            ("RandomOverSampler", RandomOverSampler()),
            ("SMOTE", SMOTE())]
        self.pca_components_list = config.PCA_COMPONENTS_LIST
        tools.log_main(f'  - Method: {self.method}', save_path=self.run_results_path)

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

    def get_X_y_data(self):
        X = self.data.drop(self.target_col, axis=1).copy()
        X = X.fillna(0) 
        y = self.data[self.target_col].copy()
        y = y.iloc[:, 0]
        return X, y

    def basic_processing(self, X, y, PCA_cols = None):
        if PCA_cols == None:
            all_numerical_cols = self.numerical_cols + self.dos_cols
        else:
            all_numerical_cols = self.numerical_cols + PCA_cols
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = RobustScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_cols),
                ('num', numerical_transformer, all_numerical_cols)  
            ]
        )  
        X_preprocessed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(X_preprocessed, 'todense'):
            X_preprocessed = X_preprocessed.todense()
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
        return X_preprocessed_df, y
    
    @staticmethod
    def save_autogluon_results(performance, leaderboard, path):
        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_df.to_csv(os.path.join(path, 'leaderboard.csv'), index=False)
        with open(os.path.join(path, 'report.txt'), 'w') as f:
            f.write(f'Performance:\n{performance}')

    @staticmethod
    def tabular_train_test(X_train, X_test, y_train, y_test):
        train_data = pd.concat([pd.DataFrame(X_train).reset_index(drop=True), pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
        test_data = pd.concat([pd.DataFrame(X_test).reset_index(drop=True), pd.DataFrame(y_test).reset_index(drop=True)], axis=1)
        return train_data, test_data
    
    @staticmethod
    def split_resample_data(X, y, resampling_technique=None, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if resampling_technique!=None:
            X_train, y_train = resampling_technique.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test
    
    def apply_pca(self, n_PCA, X):
        pca = PCA(n_components=n_PCA)
        pca_columns = [f'PC{i+1}' for i in range(n_PCA)]
        X_pca_1 = pca.fit_transform(X[self.dos_cols]) 
        X_pca_2 = pd.DataFrame(X_pca_1, columns=pca_columns)
        X_pca = pd.concat([X.loc[:, :'is_magnetic'].reset_index(drop=True), X_pca_2.reset_index(drop=True)], axis=1)
        return X_pca, pca_columns

    def autogluon_run(self, X, y, method_path, resampling_technique, n_PCA=None):
        autogluon_path = os.path.join(method_path, 'Autogluon')
        X_train, X_test, y_train, y_test = self.split_resample_data(X, y, resampling_technique=resampling_technique, test_size=0.2, random_state=42)
        train_data, test_data = self.tabular_train_test(X_train, X_test, y_train, y_test)
        print('\n ···················· AUTOGLUON ····················\n')
        predictor = TabularPredictor(label=self.target_column, problem_type='binary', eval_metric=self.eval_metric, path=autogluon_path
                                     ).fit(TabularDataset(train_data), presets='medium_quality')
        performance = predictor.evaluate(TabularDataset(test_data))
        leaderboard = predictor.leaderboard(TabularDataset(test_data), extra_metrics=['accuracy', 'roc_auc', 'precision', 'recall'], silent=True)
        self.save_autogluon_results(performance, leaderboard, method_path)
        print('\n ···················································\n')
        tools.log_main(f'    Resampling {resampling_technique} - PCA {n_PCA}: Performance {performance}', save_path=self.run_results_path)

    def autogluon_training_workflow(self):
        self.process_data()
        self.define_columns()
        if self.method == 'Basic_Autogluon':
            X, y = self.get_X_y_data()
            X, y = self.basic_processing(X, y)
            autogluon_path = os.path.join(self.run_results_path, self.method)
            self.autogluon_run(X, y, autogluon_path, None)
        if self.method == 'Resampling_Autogluon':
            X, y = self.get_X_y_data()
            X, y = self.basic_processing(X, y)
            for technique_name, technique in self.resampling_techniques:
                autogluon_path = os.path.join(self.run_results_path, self.method, technique_name)
                self.autogluon_run(X, y, autogluon_path, technique)
        if self.method == 'PCA_Resampling_Autogluon':
            for n_PCA in self.pca_components_list:
                X, y = self.get_X_y_data()
                X, pca_columns = self.apply_pca(n_PCA, X)
                X, y = self.basic_processing(X, y, pca_columns)
                for technique_name, technique in self.resampling_techniques:
                    autogluon_path = os.path.join(self.run_results_path, self.method, str(n_PCA), technique_name)
                    self.autogluon_run(X, y, autogluon_path, technique, n_PCA=n_PCA)