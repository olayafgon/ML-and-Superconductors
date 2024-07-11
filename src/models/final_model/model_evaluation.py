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
        shap_values = explainer(X_test, check_additivity=False)
        plt.figure()
        fig = shap.plots.bar(shap_values, max_display=13, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'shap_barras.png'))
        plt.figure()
        fig = shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.tight_layout()
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
    
    @staticmethod
    def plot_corr_matrix(data1, data2, label1, label2, title, save_path, save_name):
        data1_subset = data1[['fermi_energy', 'is_magnetic', 'critical_temperature_k', 'synth_doped']].copy()
        data2_subset = data2[['fermi_energy', 'is_magnetic', 'critical_temperature_k', 'synth_doped']].copy()
        corr_matrix_data1 = data1_subset.corr()
        corr_matrix_data2 = data2_subset.corr()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(corr_matrix_data1, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
        cbar = ax1.figure.colorbar(im1, ax=ax1)
        ax1.set_xticks(np.arange(len(corr_matrix_data1.columns)))
        ax1.set_yticks(np.arange(len(corr_matrix_data1.columns)))
        ax1.set_xticklabels(corr_matrix_data1.columns, rotation=45, ha="right")
        ax1.set_yticklabels(corr_matrix_data1.columns)
        for i in range(len(corr_matrix_data1.columns)):
            for j in range(len(corr_matrix_data1.columns)):
                text = ax1.text(j, i, f"{corr_matrix_data1.iloc[i, j]:.2f}", ha="center", va="center", color="black", size=8)
        ax1.set_title(f'Correlation Matrix - {label1}')
        im2 = ax2.imshow(corr_matrix_data2, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
        cbar = ax2.figure.colorbar(im2, ax=ax2)
        ax2.set_xticks(np.arange(len(corr_matrix_data2.columns)))
        ax2.set_yticks(np.arange(len(corr_matrix_data2.columns)))
        ax2.set_xticklabels(corr_matrix_data2.columns, rotation=45, ha="right")
        ax2.set_yticklabels(corr_matrix_data2.columns)
        for i in range(len(corr_matrix_data2.columns)):
            for j in range(len(corr_matrix_data2.columns)):
                text = ax2.text(j, i, f"{corr_matrix_data2.iloc[i, j]:.2f}", ha="center", va="center", color="black", size=8)
        ax2.set_title(f'Correlation Matrix - {label2}')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name + '.png'))
        plt.close()
    
    @staticmethod
    def plot_median_dos_superposed_4(data_materials1, data_materials2, data_materials3, data_materials4, title1, title2, title3, title4, combined_title, save_path, save_name):
        data_plot1 = data_materials1.copy()
        data_plot2 = data_materials2.copy()
        data_plot3 = data_materials3.copy()
        data_plot4 = data_materials4.copy()
        plt.figure(figsize=(8,4))
        average_dos1 = data_plot1.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)  
        average_dos2 = data_plot2.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)  
        average_dos3 = data_plot3.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)  
        average_dos4 = data_plot4.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)
        plt.plot(average_dos1, label=title1)
        plt.plot(average_dos2, label=title2)
        plt.plot(average_dos3, label=title3)
        plt.plot(average_dos4, label=title4)
        num_cols = len(data_materials1.columns)
        center_index = num_cols // 2 
        plt.axhline(y=0, color='black', linestyle=':')
        plt.axvline(x=center_index, color='black', linestyle=':') 
        plt.xticks([0, center_index, num_cols - 1], labels=['$E_f-15eV$', '0', '$E_f+15eV$'])
        plt.ylabel("Density of States (DOS)")
        plt.xlabel("Energía - Energía de Fermi")
        plt.title(combined_title)
        plt.grid(False)
        plt.legend()
        plt.ylim(-0.5, 8.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, save_name + '.png'))
        plt.close()
        
    @staticmethod
    def plot_median_dos_superposed_2(data_materials1, data_materials2, title1, title2, combined_title, save_path, save_name):
        data_plot1 = data_materials1.copy()
        data_plot2 = data_materials2.copy()
        plt.figure(figsize=(8,4))
        average_dos1 = data_plot1.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)  
        average_dos2 = data_plot2.loc[:, 'DOS_m15_00':'DOS_p15_00'].median(axis=0)  
        plt.plot(average_dos1, label=title1)
        plt.plot(average_dos2, label=title2)
        num_cols = len(data_materials1.columns)
        center_index = num_cols // 2 
        plt.axhline(y=0, color='black', linestyle=':')
        plt.axvline(x=center_index, color='black', linestyle=':') 
        plt.xticks([0, center_index, num_cols - 1], labels=['$E_f-15eV$', '0', '$E_f+15eV$'])
        plt.ylabel("Density of States (DOS)")
        plt.xlabel("Energía - Energía de Fermi")
        plt.title(combined_title)
        plt.grid(False)
        plt.legend()
        plt.ylim(-0.5, 8.5)
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
        self.plot_shap_values(self.model, self.X_test, figures_save_path)
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
        self.plot_corr_matrix(self.true_positive, self.false_negative, 'Verdaderos positivos', 'Falsos negativos', 
                                        'Correlaciones en predicciones de superconductores', 
                                        figures_save_path, 'correlations_supercond_predictions')
        self.plot_corr_matrix(self.false_positive, self.true_negative, 'Falsos positivos', 'Verdaderos negativos', 
                                        'Correlaciones en predicciones de no superconductores', 
                                        figures_save_path, 'correlations_non_supercond_predictions')
        self.plot_median_dos_superposed_4(self.true_positive, self.false_negative, self.true_negative, self.false_positive,  
                                        'Mediana verdaderos positivos', 'Mediana falsos negativos', 
                                        'Mediana verdaderos negativos', 'Mediana falsos positivos',
                                        'Medianas de la DOS',
                                        figures_save_path, 'dos_median_all_predictions')
        self.plot_median_dos_superposed_2(self.true_positive, self.false_negative, 'Verdaderos positivos', 'Falsos negativos', 
                                        'Medianas de la DOS en predicciones de superconductores', 
                                        figures_save_path, 'dos_median_supercond_predictions')
        self.plot_median_dos_superposed_2(self.false_positive, self.true_negative, 'Falsos positivos', 'Verdaderos negativos',  
                                        'Medianas de la DOS en predicciones de no superconductores', 
                                        figures_save_path, 'dos_median_non_supercond_predictions')


    def model_evaluation_run(self):
        self.evaluate()
        self.calculate_save_metrics()
        self.get_test_data_and_pred()
        self.get_negatives_positives()
        self.plot_figures()
        
        