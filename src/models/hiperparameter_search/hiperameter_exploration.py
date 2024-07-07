import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

pd.set_option('display.max_columns', 100)

sys.path.append('./../../')
import config
from utils import tools

class HiperparameterExploration:
    """
    A class for performing hyperparameter exploration using Bayesian and Random Search for XGBoost and LightGBM classifiers.

    Attributes:
        run_results_path (str): The path to save the run results.
        method (str): The hyperparameter exploration method to use.
        Data_Processor (DataPreprocessor): The data preprocessor object.
        eval_metric (str): The evaluation metric to use.
        n_iter (int): The number of iterations for hyperparameter search.
        n_pca (int): The number of PCA components to use.
        n_cv (int): The number of cross-validation folds.
        search_space_bayes (dict): The hyperparameter search space for Bayesian Search.
        search_space_random (dict): The hyperparameter search space for Random Search.
    """
    def __init__(self, run_results_path, method, data_processor):
        """
        Constructs all the necessary attributes for the HiperparameterExploration object.

        ARGS:
            run_results_path (str): The path to save the run results.
            method (str): The hyperparameter exploration method to use.
            data_processor (DataPreprocessor): The data preprocessor object.
        """
        self.run_results_path = run_results_path
        self.method = method
        self.data_processor = data_processor
        self.eval_metric = config.EVAL_METRIC
        self.n_iter = config.HIPERPARAMETER_EXPLO_ITER
        self.n_pca = config.HIPERPARAMETER_EXPLO_PCA
        self.n_cv = config.HIPERPARAMETER_EXPLO_CV
        self.search_space_bayes = {
            'learning_rate': Real(0.01, 1.0),   # Learning rate
            'max_depth': Integer(3, 30),        # Maximum tree depth
            'n_estimators': Integer(50, 1500),  # Number of trees
            'subsample': Real(0.5, 1),          # Subsample ratio of training instances
            'colsample_bytree': Real(0.5, 1)   # Subsample ratio of columns
            }
        self.search_space_random = {
            'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0],
            'max_depth': [3, 5, 10, 15, 20, 30],
            'n_estimators': [50, 100, 200, 500, 700, 1000, 1500],
            'subsample': [0.01, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0],
            'colsample_bytree': [0.01, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
            }
        tools.log_main(f'  - Method: {self.method}', save_path=self.run_results_path)
    
    @staticmethod
    def split_resample_data(X, y, resampling_technique=None, test_size=0.2, random_state=42):
        """
        Splits the data into training and test sets and applies resampling if specified.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            resampling_technique (object, optional): The resampling technique to use. Defaults to None.
            test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.
            random_state (int, optional): The random seed to use for splitting the data. Defaults to 42.

        RETURNS:
            tuple: A tuple containing the training and test features and target.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if resampling_technique!=None:
            X_train, y_train = resampling_technique.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def run_search_bayes(X, y, model_class, name, search_space, n_iter = 100, scoring = 'f1', cv=5):
        """
        Performs Bayesian Optimization for hyperparameter tuning using the given model and search space.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            model_class (class): The machine learning model class to use (e.g., XGBClassifier).
            name (str): The name of the search method (for reporting).
            search_space (dict): The hyperparameter search space for Bayesian Search.
            n_iter (int, optional): The number of iterations for hyperparameter search. Defaults to 100.
            scoring (str, optional): The evaluation metric to use for scoring. Defaults to 'f1'.
            cv (int, optional): The number of cross-validation folds. Defaults to 5.

        RETURNS:
            object: The fitted BayesSearchCV object.
        """
        search_method = BayesSearchCV(
            estimator=model_class(),
            search_spaces=search_space,
            n_jobs=-1,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring,
            refit=True,
            verbose=1
        )
        print(f"\n--- {name} ---")
        search_method.fit(X, y)
        return search_method

    @staticmethod
    def run_search_random(X, y, model_class, name, search_space, n_iter = 100, scoring = 'f1', cv=5):
        """
        Performs Random Search for hyperparameter tuning using the given model and search space.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            model_class (class): The machine learning model class to use (e.g., LGBMClassifier).
            name (str): The name of the search method (for reporting).
            search_space (dict): The hyperparameter search space for Random Search.
            n_iter (int, optional): The number of iterations for hyperparameter search. Defaults to 100.
            scoring (str, optional): The evaluation metric to use for scoring. Defaults to 'f1'.
            cv (int, optional): The number of cross-validation folds. Defaults to 5.

        RETURNS:
            object: The fitted RandomizedSearchCV object.
        """
        search_method = RandomizedSearchCV(
            estimator=model_class(),
            param_distributions=search_space,
            n_jobs=-1,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring,
            refit=True,
            verbose=1
        )
        print(f"\n--- {name} ---")
        search_method.fit(X, y)
        return search_method

    @staticmethod
    def report_eval_metrics(best_param, model, X_test, y_test, name, report_path):
        """
        Evaluates the given model on the test set and reports the results.

        ARGS:
            best_param (dict): The best hyperparameters found during the search.
            model (object): The fitted machine learning model.
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test target.
            name (str): The name of the search method used (for reporting).
            report_path (str): The directory to save the report.
        """
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        confusion_matrix_result = confusion_matrix(y_test, y_pred)

        with open(os.path.join(report_path, 'hiperparameter_methods_results.txt'), 'a') as file:
            file.write(f"\n--- {name} ---\n")
            file.write(f"Mejores hiperparámetros: {best_param}\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1-score: {f1:.4f}\n")
            file.write(f"AUC: {auc:.4f}\n")
            file.write(f"Matriz de confusión:\n{confusion_matrix_result}\n")

    def run_search(self, X_train, y_train):
        """
        Executes the selected hyperparameter search methods (Random and Bayesian) for both XGBoost and LightGBM.

        ARGS:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target.
        """
        self.random_search_xgb = self.run_search_random(X_train, y_train, XGBClassifier, 'RandomizedSearchCV (XGBoost)', 
                                                        self.search_space_random, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.bayes_search_xgb = self.run_search_bayes(X_train, y_train, XGBClassifier, 'BayesSearchCV (XGBoost)', 
                                                      self.search_space_bayes, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.random_search_lgbm = self.run_search_random(X_train, y_train, LGBMClassifier, 'RandomizedSearchCV (LightGBM)', 
                                                         self.search_space_random, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.bayes_search_lgbm = self.run_search_bayes(X_train, y_train, LGBMClassifier, 'BayesSearchCV (LightGBM)', 
                                                       self.search_space_bayes, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        
    def evaluate_search_method(self, X_test, y_test):
        """
        Evaluates each search method (Random, Bayesian) for both XGBoost and LightGBM on the test set.

        ARGS:
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The test target.
        """
        for search_method, name in [(self.bayes_search_lgbm, 'BayesSearchCV (LightGBM)'), 
                                    (self.random_search_lgbm, 'RandomizedSearchCV (LightGBM)'),
                                    (self.bayes_search_xgb, 'BayesSearchCV (XGBoost)'),
                                    (self.random_search_xgb, 'RandomizedSearchCV (XGBoost)')]:
            best_model = search_method.best_estimator_
            best_param = search_method.best_params_
            report_path = os.path.join(self.run_results_path, self.method)
            self.report_eval_metrics(best_param, best_model, X_test, y_test, name, report_path)
            
    def hiperparameter_exploration_run(self):
        """
        Executes the complete hyperparameter exploration pipeline, including preprocessing, searching, and evaluating.
        """
        X, y = self.data_processor.preprocess_data()
        X, pca_columns = self.data_processor.apply_pca(self.n_pca, X)
        X, y = self.data_processor.basic_processing(X, y, pca_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.run_search(X_train, y_train)
        self.evaluate_search_method(X_test, y_test)