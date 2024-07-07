import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', 100)

sys.path.append('./../../')
import config
from utils import tools

class AutogluonTraining:
    """
    A class to train and evaluate machine learning models using AutoGluon.

    Attributes:
        run_results_path (str): The path to save the run results.
        method (str): The AutoGluon training method to use.
        Data_Processor (DataPreprocessor): The data preprocessor object.
        target_column (str): The name of the target column.
        eval_metric (str): The evaluation metric to use.
        resampling_techniques (list): A list of resampling techniques to use.
        pca_components_list (list): A list of PCA components to use.
    """
    def __init__(self, run_results_path, method, Data_Processor):
        """
        Constructs all the necessary attributes for the AutogluonTraining object.

        ARGS:
            run_results_path (str): The path to save the run results.
            method (str): The AutoGluon training method to use.
            Data_Processor (DataPreprocessor): The data preprocessor object.
        """
        self.run_results_path = run_results_path
        self.method = method
        self.Data_Processor = Data_Processor
        self.target_column = config.TARGET_COLUMN
        self.eval_metric = config.EVAL_METRIC
        self.resampling_techniques = [
            ("none", None),
            ("RandomUnderSampler", RandomUnderSampler()),
            ("RandomOverSampler", RandomOverSampler()),
            ("SMOTE", SMOTE())]
        self.pca_components_list = config.PCA_COMPONENTS_LIST
        tools.log_main(f'  - Method: {self.method}', save_path=self.run_results_path)
    
    @staticmethod
    def save_autogluon_results(performance, leaderboard, path):
        """
        Saves the AutoGluon results to a file.

        ARGS:
            performance (dict): The model performance metrics.
            leaderboard (pd.DataFrame): The model leaderboard.
            path (str): The path to save the results.
        """
        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_df.to_csv(os.path.join(path, 'leaderboard.csv'), index=False)
        with open(os.path.join(path, 'report.txt'), 'w') as f:
            f.write(f'Performance:\n{performance}')

    @staticmethod
    def tabular_train_test(X_train, X_test, y_train, y_test):
        """
        Combines the features and target data into a single DataFrame for AutoGluon.

        ARGS:
            X_train (pd.DataFrame): The training features.
            X_test (pd.DataFrame): The test features.
            y_train (pd.Series): The training target.
            y_test (pd.Series): The test target.

        RETURNS:
            tuple: A tuple containing the training and test DataFrames.
        """
        train_data = pd.concat([pd.DataFrame(X_train).reset_index(drop=True), pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
        test_data = pd.concat([pd.DataFrame(X_test).reset_index(drop=True), pd.DataFrame(y_test).reset_index(drop=True)], axis=1)
        return train_data, test_data
    
    @staticmethod
    def split_resample_data(X, y, resampling_technique=None, test_size=0.2, random_state=42):
        """
        Splits the data into training and test sets and applies resampling if specified.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            resampling_technique (object, optional): The resampling technique to use. 
                                                     Defaults to None.
            test_size (float, optional): The proportion of the data to use for testing. 
                                         Defaults to 0.2.
            random_state (int, optional): The random seed to use for splitting the data. 
                                          Defaults to 42.

        RETURNS:
            tuple: A tuple containing the training and test features and target.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if resampling_technique!=None:
            X_train, y_train = resampling_technique.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test
    
    def autogluon_run(self, X, y, method_path, resampling_technique, n_PCA=None):
        """
        Trains and evaluates an AutoGluon model with the specified parameters.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            method_path (str): The path to save the method results.
            resampling_technique (object, optional): The resampling technique used. 
                                                     Defaults to None.
            n_PCA (int, optional): The number of PCA components used. Defaults to None.
        """
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
        """
        Executes the AutoGluon training workflow based on the chosen method.
        """
        X, y = self.Data_Processor.preprocess_data()
        if self.method == 'Basic_Autogluon':
            X, y = self.Data_Processor.basic_processing(X, y)
            autogluon_path = os.path.join(self.run_results_path, self.method)
            self.autogluon_run(X, y, autogluon_path, None)
        if self.method == 'Resampling_Autogluon':
            X, y = self.Data_Processor.basic_processing(X, y)
            for technique_name, technique in self.resampling_techniques:
                autogluon_path = os.path.join(self.run_results_path, self.method, technique_name)
                self.autogluon_run(X, y, autogluon_path, technique)
        if self.method == 'PCA_Resampling_Autogluon':
            for n_PCA in self.pca_components_list:
                X, pca_columns = self.Data_Processor.apply_pca(n_PCA, X)
                X, y = self.Data_Processor.basic_processing(X, y, pca_columns)
                for technique_name, technique in self.resampling_techniques:
                    autogluon_path = os.path.join(self.run_results_path, self.method, str(n_PCA), technique_name)
                    self.autogluon_run(X, y, autogluon_path, technique, n_PCA=n_PCA)