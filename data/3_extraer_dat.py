import os
import matplotlib.pyplot as plt
import numpy as np

def read_doscar(file_path, output_file_path):
    with open(file_path, 'r') as file:
        # Leer las líneas relevantes del archivo
        lines = file.readlines()

        # Obtener la información necesaria
        energy_range = [float(value) for value in lines[5].split()[2:4]]
        num_dos_points = int(float(lines[5].split()[4]))  # Convertir a coma flotante primero y luego a entero

        # Encontrar la última fila con 3 columnas
        last_valid_row = None
        for i, line in enumerate(lines[6:]):
            if len(line.split()) == 3:
                last_valid_row = i + 6  # Ajustar el índice para tener en cuenta las líneas iniciales

        # Leer todas las columnas de energía y DOS hasta la última fila válida
        data = np.loadtxt(lines[6:last_valid_row], unpack=True)

    # Guardar todos los datos en un archivo .dat
    with open(output_file_path, 'w') as output_file:
        np.savetxt(output_file, data.T)

    return energy_range, num_dos_points, data


# Ejemplo de uso
file_path = r"D:\Project\data\Final_Data\Data_raw\Data_BCC\BCC.Ag1F6Sb1_ICSD_28676"
output_file_path = r"D:\Project\data\Final_Data\BCC.Ag1F6Sb1_ICSD_28676.dat"

read_doscar(file_path, output_file_path)
