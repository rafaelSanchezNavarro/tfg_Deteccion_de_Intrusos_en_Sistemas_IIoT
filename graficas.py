import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# Datos de edades
edades = np.random.randint(1, 9, 20).reshape(-1, 1)
edades = np.vstack([[0], edades, [10]])

# Aplicar discretización usando KBinsDiscretizer con 3 intervalos
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
edades_discretizadas = discretizer.fit_transform(edades)

# Obtener los límites de los intervalos
intervalos = discretizer.bin_edges_[0]

# Graficar los datos originales y los intervalos de discretización
plt.figure(figsize=(8, 5))

# Colores para los intervalos
colores = ['#DAE8FC', '#F8CECC', '#D5E8D4']

# Resaltar los intervalos
plt.axhspan(intervalos[0], intervalos[1], color=colores[0], alpha=0.5, label="Intervalo 1")
plt.axhspan(intervalos[1], intervalos[2], color=colores[1], alpha=0.5, label="Intervalo 2")
plt.axhspan(intervalos[2], intervalos[3], color=colores[2], alpha=0.5, label="Intervalo 3")

# Asignar colores a los puntos según el intervalo
for i, edad in enumerate(edades):
    if edad <= intervalos[1]:
        plt.scatter(i, edad, color=colores[0], edgecolor='grey')
    elif edad <= intervalos[2]:
        plt.scatter(i, edad, color=colores[1], edgecolor='grey')
    else:
        plt.scatter(i, edad, color=colores[2], edgecolor='grey')

# Dibujar los límites de los intervalos
for i in range(1, len(intervalos)-1):
    plt.axhline(y=intervalos[i], color='grey', linestyle='--')

# Mejorar el índice en el eje X
plt.xticks(range(len(edades)), edades.flatten(), rotation=45)

# Ajustar los límites de los ejes
plt.ylim(0, 10)  # Limitar el eje Y de 0 a 100

# Mostrar solo los intervalos en la leyenda
plt.legend(loc='upper left')
plt.title("Discretización con intervalos")
plt.tight_layout()
plt.show()
