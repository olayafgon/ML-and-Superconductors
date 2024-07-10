import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
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
        
    @staticmethod
    def plot_fermi_distribution_compare(data1, data2, label1, label2, title, save_path, save_name):
        plt.figure(figsize=(8, 4))
        sns.histplot(data1['fermi_energy'], kde=True, color='#66C2A5', label=label1, stat='density')
        sns.histplot(data2['fermi_energy'], kde=True, color='#FC8D62', label=label2, stat='density')
        plt.legend()
        plt.title(title)
        plt.xlabel('Energía de Fermi')
        plt.ylabel('Densidad')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name+'.png'))
        plt.close()
    
    @staticmethod
    def plot_magnetism_distribution(data1, data2, label1, label2, title, save_path, save_name):
        data1_counts = data1['is_magnetic'].value_counts().sort_index().reindex([0, 1], fill_value=0)
        data2_counts = data2['is_magnetic'].value_counts().sort_index().reindex([0, 1], fill_value=0)
        total_counts1 = data1_counts.sum()
        total_counts2 = data2_counts.sum()

        labels = ['No', 'Sí']
        x = range(len(labels))
        bar_width = 0.35
        plt.figure(figsize=(8, 4))
        bars1 = plt.bar(x, data1_counts, width=bar_width, color='#66C2A5', label=label1)
        bars2 = plt.bar([p + bar_width for p in x], data2_counts, width=bar_width, color='#FC8D62', label=label2)
        max_height = max(max(data1_counts), max(data2_counts))
        for bar, count in zip(bars1, data1_counts):
            yval = bar.get_height()
            percent = yval / total_counts1 * 100
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval)}\n({percent:.1f}%)', va='bottom', ha='center')
        for bar, count in zip(bars2, data2_counts):
            yval = bar.get_height()
            percent = yval / total_counts2 * 100
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{int(yval)}\n({percent:.1f}%)', va='bottom', ha='center')

        plt.ylim(0, max_height * 1.25)
        plt.xlabel('¿Es magnético?')
        plt.ylabel('Frecuencia')
        plt.title(title)
        plt.xticks([p + bar_width / 2 for p in x], labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name + '.png'))
        plt.close()

    @staticmethod
    def plot_bravais_lattice_distribution(data1, data2, label1, label2, title, save_path, save_name):
        combined_categories = np.unique(data1['bravais_lattice'].values.tolist() + data2['bravais_lattice'].values.tolist())
        counts1 = data1['bravais_lattice'].value_counts().reindex(combined_categories, fill_value=0)
        counts2 = data2['bravais_lattice'].value_counts().reindex(combined_categories, fill_value=0)
        bar_width = 0.35
        x = np.arange(len(combined_categories))
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(111)
        ax.bar(x - bar_width/2, counts1, width=bar_width, color='#66C2A5', align='center', label=label1)
        ax.bar(x + bar_width/2, counts2, width=bar_width, color='#FC8D62', align='center', label=label2)
        ax.set_title(title)
        ax.set_xlabel('Red de Bravais')
        ax.set_ylabel('Frecuencia')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_categories, rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name + '.png'))
        plt.close()
        
    def get_test_data_and_pred(self):
        self.data_test['predicted_superconductor'] = self.y_pred.astype(bool).copy()
        self.data_test['ICSD'] = self.data_test['ICSD'].astype('Int64')
        self.supercon_data['ICSD'] = self.supercon_data['ICSD'].astype('Int64')
        self.test_data_and_pred = pd.merge(self.data_test, self.supercon_data[['ICSD', 'critical_temperature_k', 'synth_doped']], on='ICSD', how='left')
    
    def get_negatives_positives(self):
        self.true_positive = self.test_data_and_pred[(self.test_data_and_pred['is_superconductor'] == True) & (self.test_data_and_pred['predicted_superconductor'] == True)].copy()
        self.false_negative = self.test_data_and_pred[(self.test_data_and_pred['is_superconductor'] == True) & (self.test_data_and_pred['predicted_superconductor'] == False)].copy()
        self.false_positive = self.test_data_and_pred[(self.test_data_and_pred['is_superconductor'] == False) & (self.test_data_and_pred['predicted_superconductor'] == True)].copy()
        self.true_negative = self.test_data_and_pred[(self.test_data_and_pred['is_superconductor'] == False) & (self.test_data_and_pred['predicted_superconductor'] == False)].copy()
        
    def plot_figures(self):
        figures_save_path = os.path.join(self.eval_folder_path, 'plots')
        tools.create_folder(figures_save_path)
        if self.model_algorithm == 'XGBClassifier':
            self.plot_training_curve(self.model, figures_save_path)
        # self.plot_shap_values(self.model, self.X_test, figures_save_path)
        self.plot_fermi_distribution_compare(self.true_positive, self.false_negative, 'Verdaderos positivos', 'Falsos negativos', 
                                            'Energía de Fermi en predicciones de superconductores', 
                                            figures_save_path, 'fermi_energy_supercond_predictions')
        self.plot_fermi_distribution_compare(self.false_positive, self.true_negative, 'Falsos positivos', 'Verdaderos negativos', 
                                            'Energía de Fermi en predicciones de no superconductores', 
                                            figures_save_path, 'fermi_energy_non_supercond_predictions')
        self.plot_magnetism_distribution(self.true_positive, self.false_negative, 'Verdaderos positivos', 'Falsos negativos', 
                                        'Magnetismo en predicciones de superconductores', 
                                        figures_save_path, 'magnetism_supercond_predictions')
        self.plot_magnetism_distribution(self.false_positive, self.true_negative, 'Falsos positivos', 'Verdaderos negativos',
                                        'Magnetismo en predicciones de no superconductores', 
                                        figures_save_path, 'magnetism_supercond_non_predictions')
        self.plot_bravais_lattice_distribution(self.true_positive, self.false_negative, 'Verdaderos positivos', 'Falsos negativos', 
                                        'Red de Bravais en predicciones de superconductores', 
                                        figures_save_path, 'bravais_lattice_supercond_predictions')
        self.plot_bravais_lattice_distribution(self.false_positive, self.true_negative, 'Falsos positivos', 'Verdaderos negativos', 
                                        'Red de Bravais en predicciones de no superconductores', 
                                        figures_save_path, 'bravais_lattice_non_supercond_predictions')

    def model_evaluation_run(self):
        self.evaluate()
        self.calculate_save_metrics()
        self.get_test_data_and_pred()
        self.get_negatives_positives()
        self.plot_figures()
        
        