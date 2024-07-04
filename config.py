# ARCHIVO DE CONFIGURACIÓN DEL PROYECTO

# ··················· GENERAL ···················
# Ruta relativa a la carpeta donde se guardan los resultados y logs
RESULTS_FOLDER = r'.\..\results'
# Rutas a los datos
DATA_FOLDER_PATH = r'.\..\data' 
DOS_CSV_PATH = r'.\..\data\dos_data.csv'
SUPERCON_CSV_PATH = r'.\..\data\3DSC\3DSC_ICSD_only_IDs.csv'
## Test
PATH_TEST_FIGURES = r'.\..\results\test' 

# ··················· OPCIONES ···················
# True si se quieren descargar los datos, False si ya están descargados (deberán estar en DATA_FOLDER_PATH)
DATA_DOWNLOAD = False 
# True si se quieren leer los datos de data_raw y guardarlos en csv, False si quieres leer los guardados en data
READ_DATA_RAW = False 

# ··················· DESCARGA ···················
# Velocidad límite de descompresión
SPEED_LIMIT = 2 * 500 * 1024 * 1024 # 2*500Mb/s
# Redes de Bravais sobre las que se desea trabajar
STRUCTURES = ["BCC", "BCT", "CUB", "FCC", "HEX", "MCL", "MCLC", "ORC", "ORCC", "ORCF", "ORCI", "RHL", "TET", "TRI"] 
# Tipo de archivo en Aflowlib ('DOSCAR.static.xz' o '_dosdata.json.xz')
DATA_FILE_TYPE = '_dosdata.json.xz'

# ··················· PROCESADO ···················
# Limites de energía respecto a Fermi en la lectura
EFERMI_LIMIT = 15
EFERMI_GRID_POINTS = 1999 # Esto funciona para limite en 15, recalcular para otros limites

# ··················· EDA ···················
# True si quieres que se ejecute el EDA
RUN_EDA = True
# True si quieres que se guarden representaciones relativas a las DOS
DOS_PLOTS = True
# Lista de materiales a plotear la DOS, si no None
ICSD_PLOTS = [189400, 608582, 609426]

# ··················· MODELO ···················
# Incluya en la lista los modelos que quiera explorar:
#   - 'Basic_Autogluon': Autogluon sin procesados especiales
#   - 'Resampling_Autogluon': Autogluon para varios remuestreos
#   - 'PCA_Resampling_Autogluon': Autogluon para varios remuestreos y PCAs (largo tiempo)
#   - 'Hiperparameters': Búsqueda de mejores hiperparámetros
#   - None: No explora nada de lo anterior
MODEL_EXPLORATION = ['Basic_Autogluon', 'Resampling_Autogluon', 'PCA_Resampling_Autogluon', 'Hiperparameters']
# True si quieres entrenar un modelo final
FINAL_MODEL_TRAINING = True
# Modelo a entrenar. Implementados:
#   - 'XGBClassifier'
#   - 'LGBMClassifier'
FINAL_MODEL = 'XGBClassifier'
# Hiperparámetros:
#   - 'Default': por defecto (solo para XGBClassifier) (FINAL_MODEL_DEFAULT_HIPERPARAMETERS)
#   - 'Searched': hiperparámetros para MODEL_EXPLORATION = ['Hiperparameters']
#   - 'Custom': incluye tus parámetros (FINAL_MODEL_CUSTOM_HIPERPARAMETERS)
FINAL_MODEL_HIPERPARAMETERS = 'Default'
FINAL_MODEL_DEFAULT_HIPERPARAMETERS = {
    'colsample_bytree': 0.5,
    'gamma': 0.01,
    'learning_rate': 0.22346832437911052,
    'max_depth': 24,
    'n_estimators': 1500,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'subsample': 0.8240736137234501
}
FINAL_MODEL_CUSTOM_HIPERPARAMETERS = None
# None si no quieres usar PCAs, int si quieres usarlos
PCA_NUMBER = 10

