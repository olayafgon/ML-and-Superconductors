# ARCHIVO DE CONFIGURACIÓN DEL PROYECTO

# GENERAL
RESULTS_FOLDER = r'.\..\results' # Ruta relativa a la carpeta donde se guardan los resultados y logs

# DESCARGA DE DATOS DE AFLOWLIB
DATA_DOWNLOAD = True # True si se quieren descargar los datos, False si ya están descargados (deberán estar en DATA_FOLDER_PATH)
DATA_FOLDER_PATH = r'.\..\data' # Ruta relativa a la carpeta donde se guardan los datos
STRUCTURES = ["BCC"] # Redes de Bravais sobre las que se desea trabajar
# ["BCC", "BCT", "CUB", "FCC", "HEX", "MCL", "MCLC", "ORC", "ORCC", "ORCF", "ORCI", "RHL", "TET", "TRI"]
DATA_FILE_TYPE = 'DOSCAR.static.xz' # Tipo de archivo tal como aparece en Aflowlib