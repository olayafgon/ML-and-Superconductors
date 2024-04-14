# ARCHIVO DE CONFIGURACIÓN DEL PROYECTO

# GENERAL
RESULTS_FOLDER = r'.\..\results' # Ruta relativa a la carpeta donde se guardan los resultados y logs

# DESCARGA DE DATOS DE AFLOWLIB
DATA_DOWNLOAD = False # True si se quieren descargar los datos, False si ya están descargados (deberán estar en DATA_FOLDER_PATH)
DATA_FOLDER_PATH = r'.\..\data' # Ruta relativa a la carpeta donde se guardan los datos
STRUCTURES = ["FCC", "HEX", "MCL", "MCLC", "ORC", "ORCC", "ORCF", "ORCI", "RHL", "TET", "TRI"] # Redes de Bravais sobre las que se desea trabajar
# "BCC", "BCT", "CUB", 
# Tipo de archivo en Aflowlib ('DOSCAR.static.xz' o '_dosdata.json.xz')
DATA_FILE_TYPE = '_dosdata.json.xz'