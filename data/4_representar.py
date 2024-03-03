import os
import matplotlib.pyplot as plt
import numpy as np

def read_doscar(file_path):
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

        # Leer las columnas de energía y DOS hasta la última fila válida
        data = np.loadtxt(lines[6:last_valid_row], unpack=True)

        energy = data[0]
        dos = data[1]

    return energy_range, num_dos_points, energy, dos


def plot_dos(energy, dos, material_name):
    plt.figure(figsize=(8, 6))
    
    # Encuentra los índices donde la DOS es diferente de cero
    nonzero_indices = np.nonzero(dos)[0]
    
    # Ajusta los límites del eje x para excluir las regiones con y=0 en los extremos
    x_min = energy[nonzero_indices[0]]-1
    x_max = energy[nonzero_indices[-1]]+1
    
    plt.plot(energy, dos, label='DOS')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    plt.title(material_name)
    plt.xlabel('Energía')
    plt.ylabel('Densidad de Estados (DOS)')
    
    # Establece los límites del eje x para excluir las regiones con y=0 en los extremos
    plt.xlim(x_min, x_max)
    
    plt.legend()
    plt.show()

# Ruta al archivo DOSCAR
doscar_file_path = r"D:\Project\data\Final_Data\Data_raw\Data_BCC\BCC.Ag1F6Sb1_ICSD_28676"
material_name = os.path.basename(doscar_file_path)

# Leer datos del archivo DOSCAR
energy_range, num_dos_points, energy, dos = read_doscar(doscar_file_path)

# Imprimir información
print(f'Energía máxima: {energy_range[0]} eV')
print(f'Energía mínima: {energy_range[1]} eV')
print(f'Número de puntos DOS: {num_dos_points}')

# Graficar la DOS frente a la energía
plot_dos(energy, dos, material_name)

