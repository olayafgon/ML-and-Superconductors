import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

sys.path.append('./../')
import config
from utils import tools


class ModelEvaluation:
    def __init__(self, run_results_path, model, X_test, y_test):
        self.run_results_path = run_results_path
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.eval_folder_path = os.path.join(self.run_results_path, 'Final_Model')
        self.model_algorithm = config.FINAL_MODEL
        if config.FINAL_MODEL_HIPERPARAMETERS == 'Default':
            self.model_hiperparameters = config.FINAL_MODEL_DEFAULT_HIPERPARAMETERS
        elif config.FINAL_MODEL_HIPERPARAMETERS == 'Custom':
            self.model_hiperparameters = config.FINAL_MODEL_CUSTOM_HIPERPARAMETERS
        self.model_pca = config.FINAL_MODEL_PCA_NUMBER
        self.model_resampling = config.FINAL_MODEL_RESAMPLER
        tools.create_folder(self.eval_folder_path)
        
    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        
    def calculate_save_metrics(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        confusion_matrix_result = confusion_matrix(self.y_test, self.y_pred)
        with open(os.path.join(self.eval_folder_path, 'report_metrics.txt'), 'a') as file:
            file.write(f"Modelo entrenado: {self.model_algorithm}\n")
            file.write(f"Método de resampleo: {self.model_resampling}\n")
            file.write(f"Número de PCAs: {self.model_algorithm}\n")
            file.write(f"Hiperparámetros empleados: {self.model_hiperparameters}\n")
            file.write(f"\n··········· Métricas ···········\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1-score: {f1:.4f}\n")
            file.write(f"AUC: {auc:.4f}\n")
            file.write(f"Matriz de confusión:\n{confusion_matrix_result}\n")
        
    def model_training_run(self):
        self.evaluate()
        self.calculate_save_metrics()
        