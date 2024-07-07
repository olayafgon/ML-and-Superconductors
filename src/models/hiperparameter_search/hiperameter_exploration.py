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
    def __init__(self, run_results_path, method, Data_Processor):
        self.run_results_path = run_results_path
        self.method = method
        self.Data_Processor = Data_Processor
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if resampling_technique!=None:
            X_train, y_train = resampling_technique.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def run_search_bayes(X, y, model_class, name, search_space, n_iter = 100, scoring = 'f1', cv=5):
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
        self.random_search_xgb = self.run_search_random(X_train, y_train, XGBClassifier, 'RandomizedSearchCV (XGBoost)', 
                                                        self.search_space_random, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.bayes_search_xgb = self.run_search_bayes(X_train, y_train, XGBClassifier, 'BayesSearchCV (XGBoost)', 
                                                      self.search_space_bayes, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.random_search_lgbm = self.run_search_random(X_train, y_train, LGBMClassifier, 'RandomizedSearchCV (LightGBM)', 
                                                         self.search_space_random, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        self.bayes_search_lgbm = self.run_search_bayes(X_train, y_train, LGBMClassifier, 'BayesSearchCV (LightGBM)', 
                                                       self.search_space_bayes, n_iter = self.n_iter, scoring = self.eval_metric, cv=self.n_cv)
        
    def evaluate_search_method(self, X_test, y_test):
        for search_method, name in [(self.bayes_search_lgbm, 'BayesSearchCV (LightGBM)'), 
                                    (self.random_search_lgbm, 'RandomizedSearchCV (LightGBM)'),
                                    (self.bayes_search_xgb, 'BayesSearchCV (XGBoost)'),
                                    (self.random_search_xgb, 'RandomizedSearchCV (XGBoost)')]:
            best_model = search_method.best_estimator_
            best_param = search_method.best_params_
            report_path = os.path.join(self.run_results_path, self.method)
            self.report_eval_metrics(best_param, best_model, X_test, y_test, name, report_path)
            
    def hiperparameter_exploration_run(self):
        X, y = self.Data_Processor.preprocess_data()
        X, pca_columns = self.Data_Processor.apply_pca(self.n_pca, X)
        X, y = self.Data_Processor.basic_processing(X, y, pca_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.run_search(X_train, y_train)
        self.evaluate_search_method(X_test, y_test)