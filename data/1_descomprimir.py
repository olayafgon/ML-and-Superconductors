import os
import lzma
import shutil
import time

# Ruta al directorio que contiene los archivos comprimidos
directorio_origen = '01_Data_compressed'

# Ruta al directorio donde deseas descomprimir los archivos
directorio_destino = '02_Data_raw'

# Ruta al archivo .txt donde se guardarán los nombres de archivos que no se pueden descomprimir
archivo_error = '00_Errores.txt'

# Velocidad límite en bytes por segundo (500 MB/s)
limite_velocidad = 500 * 1024 * 1024

# Verifica si el directorio de origen existe
if not os.path.exists(directorio_origen):
    print(f'Error: El directorio de origen "{directorio_origen}" no existe.')
    exit()

# Verifica si el directorio de destino existe, y si no, créalo
if not os.path.exists(directorio_destino):
    os.makedirs(directorio_destino)

# Recorre todas las carpetas y subcarpetas del directorio de origen
for carpeta, subcarpetas, archivos in os.walk(directorio_origen):
    for archivo_comprimido in archivos:
        if archivo_comprimido.endswith('.xz'):
            # Construye las rutas de entrada y salida
            ruta_entrada = os.path.join(carpeta, archivo_comprimido)
            carpeta_destino = carpeta.replace(directorio_origen, '').lstrip(os.path.sep)
            ruta_salida = os.path.join(directorio_destino, carpeta_destino, archivo_comprimido.replace('.xz', ''))

            # Crea la carpeta de destino si no existe
            os.makedirs(os.path.join(directorio_destino, carpeta_destino), exist_ok=True)

            try:
                # Descomprime el archivo limitando la velocidad
                with lzma.open(ruta_entrada, 'rb') as f_in, open(ruta_salida, 'wb') as f_out:
                    start_time = time.time()
                    shutil.copyfileobj(f_in, f_out, length=limite_velocidad)
                    elapsed_time = time.time() - start_time
                    print(f'Descomprimido: {ruta_entrada} (Tiempo: {elapsed_time:.2f} s)')
            except Exception as e:
                with open(archivo_error, 'a') as error_file:
                    error_file.write(f'Error en {ruta_entrada}: {str(e)}\n')
                print(f'Error en {ruta_entrada}: {str(e)}')
                continue  # Salta al siguiente archivo si hay un error

print('Proceso completado.')


