import sys
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.path.append('./../../')
import config

class DataPreprocessor:
    """
    A class to preprocess the data for machine learning models.

    Attributes:
        data (pd.DataFrame): The input data.
        target_column (str): The name of the target column.
        efermi_limit (float): The Fermi energy limit to define DOS columns.
    """
    def __init__(self, data):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        ARGS:
            data (pd.DataFrame): The input data.
        """
        self.data = data
        self.target_column = config.TARGET_COLUMN
        self.efermi_limit = config.EFERMI_LIMIT

    def process_data(self):
        """
        Preprocesses the data by removing NaN values and converting the target column to boolean type.

        """
        self.data.dropna(inplace=True)
        self.data['is_superconductor'] = self.data['is_superconductor'].astype(bool)

    def define_columns(self):
        """
        Defines the DOS, categorical, numerical, and target columns.

        """
        start_col = f"DOS_m{abs(self.efermi_limit):02.0f}_00"  
        end_col = f"DOS_p{abs(self.efermi_limit):02.0f}_00"
        self.dos_cols = self.data.loc[:, start_col:end_col].columns.tolist()
        self.categorical_cols = ['bravais_lattice']
        self.numerical_cols = ['fermi_energy', 'is_magnetic']
        self.target_col = [self.target_column]

    def get_X_y_data(self):
        """
        Separates the data into features (X) and target (y).

        RETURNS:
            tuple: A tuple containing the features (X) and target (y).
        """
        X = self.data.drop(self.target_col, axis=1).copy()
        X = X.fillna(0) 
        y = self.data[self.target_col].copy()
        y = y.iloc[:, 0]
        return X, y
    
    def preprocess_data(self):
        """
        Preprocesses the data by executing the process_data, define_columns, and get_X_y_data methods.

        RETURNS:
            tuple: A tuple containing the features (X) and target (y).
        """
        self.process_data()
        self.define_columns()
        X, y = self.get_X_y_data()
        return X, y
    
    def basic_processing(self, X, y, PCA_cols = None):
        """
        Applies basic preprocessing steps to the data, including one-hot encoding for categorical features and robust scaling for numerical features.

        ARGS:
            X (pd.DataFrame): The features data.
            y (pd.Series): The target data.
            PCA_cols (list, optional): A list of PCA column names. Defaults to None.

        RETURNS:
            tuple: A tuple containing the processed features (X_preprocessed_df) and target (y).
        """
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
        """
        Applies PCA to the DOS columns.

        ARGS:
            n_PCA (int): The number of PCA components.
            X (pd.DataFrame): The features data.

        RETURNS:
            tuple: A tuple containing the PCA-transformed features (X_pca) and the list of PCA column names (pca_columns).
        """
        pca = PCA(n_components=n_PCA)
        pca_columns = [f'PC{i+1}' for i in range(n_PCA)]
        X_pca_1 = pca.fit_transform(X[self.dos_cols]) 
        X_pca_2 = pd.DataFrame(X_pca_1, columns=pca_columns)
        X_pca = pd.concat([X.loc[:, :'is_magnetic'].reset_index(drop=True), X_pca_2.reset_index(drop=True)], axis=1)
        return X_pca, pca_columns
    
    def get_original_data_test(self, X_test):
        data_test = self.data.iloc[X_test.index].copy()
        return data_test