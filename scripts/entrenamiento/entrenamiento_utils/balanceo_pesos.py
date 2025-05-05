# Suavizado logarítmico ponderado
import numpy as np
from sklearn.utils import compute_class_weight


def suavizado_logaritmico_ponderado(n_total, n_clase, alpha=1.0):
    # Este suavizado ajusta el peso de las clases basándose en el logaritmo de la cantidad total de muestras y de la cantidad de muestras por clase.
    # El parámetro alpha permite controlar el impacto del suavizado, de modo que clases con pocos ejemplos no sean penalizadas excesivamente.
    # Si alpha es grande, el suavizado tendrá menor efecto. Ideal cuando se quiere dar un mayor control sobre la influencia del suavizado.
    return np.log(n_total + 1) / (np.log(n_clase + 1) + alpha)

# Suavizado inversamente proporcional al tamaño de la clase
def suavizado_inverso(n_total, n_clase):
    # Este suavizado es inversamente proporcional al tamaño de la clase.
    # A medida que el tamaño de la clase crece, el peso asignado disminuye. Es útil cuando se desea penalizar más las clases con menos muestras.
    # Esto puede ayudar a que las clases pequeñas tengan un mayor peso, equilibrando el desequilibrio en el dataset.
    return 1 / np.log(n_clase + 1)

# Suavizado con base 2
def suavizado_base_2(n_total, n_clase):
    # Este suavizado utiliza logaritmos en base 2, lo que da una escala diferente al suavizado logarítmico estándar.
    # A diferencia del logaritmo natural (base e), el logaritmo en base 2 proporciona un suavizado más gradual y menos pronunciado.
    # Esto puede ser útil si se desea un suavizado menos agresivo.
    return np.log2(n_total + 1) / np.log2(n_clase + 1)

# Suavizado cuadrático
def suavizado_cuadratico(n_total, n_clase):
    # Este suavizado utiliza una fórmula cuadrática, dando un peso mayor a las clases con menos ejemplos.
    # A medida que el número de muestras por clase aumenta, el peso asignado disminuye cuadráticamente.
    # Esto puede ser útil si se quiere dar un mayor énfasis a las clases pequeñas, al mismo tiempo que reduce el impacto de las clases más grandes.
    return (n_total**2) / (n_clase**2 + 1)

# Suavizado normalizado
def suavizado_normalizado(n_total, n_clase):
    # Este suavizado normaliza el logaritmo de la cantidad total de muestras con el logaritmo de la cantidad de muestras por clase.
    # Esto permite equilibrar el impacto entre las clases de manera más uniforme, considerando tanto el tamaño de la clase como el tamaño total del dataset.
    # La normalización ayuda a evitar que el suavizado sea excesivo para las clases grandes o pequeñas.
    return np.log(n_total + 1) / (np.log(n_clase + 1) + np.log(n_total + 1))

def obtener_pesos_suavizados(y_train_class2):
    # 1. Obtener las clases y su cantidad de ejemplos
    class_names = np.unique(y_train_class2)  # Obtener nombres de clases
    class_counts = [np.sum(y_train_class2 == cls) for cls in class_names]  # Obtener el conteo de muestras por clase

    # 2. Calcular los pesos de clase usando 'balanced'
    weights = compute_class_weight('balanced', classes=class_names, y=y_train_class2)
    class_weights = dict(zip(class_names, weights))

    # 3. Aplicar el suavizado logarítmico sobre los pesos calculados
    n_total = len(y_train_class2)  # Número total de muestras
    class_weights_suavizado = {}

    for cls, weight in class_weights.items():
        # Suavizado logarítmico
        class_count = class_counts[np.where(class_names == cls)[0][0]]  # Número de ejemplos para esta clase
        suavizado = suavizado_inverso(n_total, class_count)  # Calcular el suavizado
        class_weights_suavizado[cls] = weight * suavizado  # Aplicar el suavizado al peso original

    # 4. Verificación (opcional)
    # print(np.unique(y_train_class2, return_counts=True))  # Mostrar la distribución de clases
    # print("Pesos sin suavizado:", class_weights)  # Pesos calculados por 'balanced'
    # print("Pesos con suavizado:", class_weights_suavizado)  # Pesos con suavizado logarítmico

    return class_weights_suavizado