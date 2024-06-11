import sys
import pandas as pd
import re

pd.set_option('display.max_columns', None)

sys.path.append('./../')

def extract_elements_from_dataframe(df):
    """Extracts elements from the 'material_name' column of the DataFrame."""
    def extract_elements(formula):
        return re.findall(r'[A-Z][a-z]?', formula)
    df['elements'] = df['material_name'].apply(extract_elements)
    return df

def create_element_dataframe(df):
    """Creates a DataFrame with element information and superconductivity."""
    element_superconductor = []
    for i, row in df.iterrows():
        for element in row['elements']:
            element_superconductor.append([element, row['is_superconductor']])
    return pd.DataFrame(element_superconductor, columns=['element', 'is_superconductor'])