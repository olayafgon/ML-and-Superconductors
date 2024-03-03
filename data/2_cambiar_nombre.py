import os

def cambiar_nombre_archivos(directorio):
    # Obtener la lista de archivos en el directorio
    carpetas = os.listdir(directorio)
    
    # Iterar sobre cada archivo en el directorio
    for carpeta in carpetas:
        archivos = os.listdir(os.path.join(directorio_a_modificar, carpeta))

        for archivo in archivos:
            # Reemplazar el punto (.) por guion bajo (_)
            nuevo_nombre = archivo.replace('.', '_') + '.txt'

            # Construir la ruta completa del archivo antiguo y nuevo
            ruta_antiguo = os.path.join(directorio, carpeta, archivo)
            ruta_nuevo = os.path.join(directorio, carpeta, nuevo_nombre)

            # Renombrar el archivo
            os.rename(ruta_antiguo, ruta_nuevo)


# Cambiar el nombre de archivos en un directorio específico
directorio_a_modificar = r"D:\Project\data\Final_Data\Data_raw"
cambiar_nombre_archivos(directorio_a_modificar)
