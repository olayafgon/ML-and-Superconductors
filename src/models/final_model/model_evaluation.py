import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap

sys.path.append('./../')
import config
from utils import tools


class ModelEvaluation:
    def __init__(self, run_results_path, model, X_test, y_test, data_test, supercon_data):
        self.run_results_path = run_results_path
        self.X_test = X_test
        self.y_test = y_test
        self.data_test = data_test
        self.supercon_data = supercon_data
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
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman'] 
        mpl.rcParams['text.usetex'] = False
        
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
    
    @staticmethod     
    def plot_training_curve(model, save_path):
        results = model.evals_result()
        epochs = len(results['validation_0']['error'])
        plt.figure(figsize=(4, 4))
        plt.plot(range(epochs), results['validation_0']['error'], label='Error de entrenamiento')
        plt.plot(range(epochs), results['validation_1']['error'], label='Error de validación')
        plt.xlabel('epochs', fontsize=11)
        plt.ylabel('Error', fontsize=11)
        plt.title('Evolución del entrenamiento', fontsize=13)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend()
        plt.savefig(os.path.join(save_path, 'curva_entrenamiento.png'))
        plt.close()
        
    @staticmethod     
    def plot_shap_values(model, X_test, save_path):
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        fig = shap.plots.bar(shap_values, max_display=13, show=False)
        plt.savefig(os.path.join(save_path, 'shap_barras.png'))
        fig = shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.savefig(os.path.join(save_path, 'shap_enjambre.png'))
        
    def get_test_data_and_pred(self):
        self.data_test['predicted_superconductor'] = self.y_pred.astype(bool).copy()
        self.data_test['ICSD'] = self.data_test['ICSD'].astype('Int64')
        self.supercon_data['ICSD'] = self.supercon_data['ICSD'].astype('Int64')
        self.test_data_and_pred = pd.merge(self.data_test, self.supercon_data[['ICSD', 'critical_temperature_k', 'synth_doped']], on='ICSD', how='left')
        
    def plot_figures(self):
        figures_save_path = os.path.join(self.eval_folder_path, 'plots')
        tools.create_folder(figures_save_path)
        if self.model_algorithm == 'XGBClassifier':
            self.plot_training_curve(self.model, figures_save_path)
        self.plot_shap_values(self.model, self.X_test, figures_save_path)
         
    def model_evaluation_run(self):
        self.evaluate()
        self.calculate_save_metrics()
        self.plot_figures()
        self.get_test_data_and_pred()
        
        