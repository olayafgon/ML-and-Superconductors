import sys
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append('./../../')
import config

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.target_column = config.TARGET_COLUMN
        self.efermi_limit = config.EFERMI_LIMIT

    def process_data(self):
        self.data.dropna(inplace=True)
        self.data['is_superconductor'] = self.data['is_superconductor'].astype(bool)

    def define_columns(self):
        start_col = f"DOS_m{abs(self.efermi_limit):02.0f}_00"  
        end_col = f"DOS_p{abs(self.efermi_limit):02.0f}_00"
        self.dos_cols = self.data.loc[:, start_col:end_col].columns.tolist()
        self.categorical_cols = ['bravais_lattice']
        self.numerical_cols = ['fermi_energy', 'is_magnetic']
        self.target_col = [self.target_column]

    def get_X_y_data(self):
        X = self.data.drop(self.target_col, axis=1).copy()
        X = X.fillna(0) 
        y = self.data[self.target_col].copy()
        y = y.iloc[:, 0]
        return X, y
    
    def preprocess_data(self):
        self.process_data()
        self.define_columns()
        X, y = self.get_X_y_data()
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

    def apply_pca(self, n_PCA, X):
        pca = PCA(n_components=n_PCA)
        pca_columns = [f'PC{i+1}' for i in range(n_PCA)]
        X_pca_1 = pca.fit_transform(X[self.dos_cols]) 
        X_pca_2 = pd.DataFrame(X_pca_1, columns=pca_columns)
        X_pca = pd.concat([X.loc[:, :'is_magnetic'].reset_index(drop=True), X_pca_2.reset_index(drop=True)], axis=1)
        return X_pca, pca_columns