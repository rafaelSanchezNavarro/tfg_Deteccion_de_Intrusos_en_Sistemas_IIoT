import pandas as pd
from sklearn.metrics import roc_auc_score

# Cargar los dos CSV en DataFrames
csv1 = pd.read_csv('predicciones/real.csv', header=None)
csv2 = pd.read_csv('predicciones/op2.csv', header=None)

# # Asegurar que ambos DataFrames tengan el mismo número de filas
# min_filas = min(len(csv1), len(csv2))
# csv1, csv2 = csv1.iloc[:min_filas, :3], csv2.iloc[:min_filas, :3]

# # Comparar filas completas (primeras 3 columnas)
# diferencias = (csv1 != csv2).any(axis=1)

# # Obtener los índices de las filas diferentes
# indices_filas_diferentes = csv1.index[diferencias].tolist()

# Asegurar que ambos DataFrames tengan el mismo número de filas y seleccionar las tres primeras columnas
min_filas = min(len(csv1), len(csv2))
y_real_ataque, y_real_categoria, y_real_tipo = csv1.iloc[:min_filas, 0], csv1.iloc[:min_filas, 1], csv1.iloc[:min_filas, 2]
y_pred_ataque, y_pred_categoria, y_pred_tipo = csv2.iloc[:min_filas, 0], csv2.iloc[:min_filas, 1], csv2.iloc[:min_filas, 2]

# Filtrar solo las filas donde el ataque (columna 0) es 1
ataque_filas = (y_real_ataque == 1)

# Definir casos en la matriz de confusión
TP = ((y_real_ataque == 1) & (y_pred_ataque == 1) & (y_real_categoria == y_pred_categoria) & (y_real_tipo == y_pred_tipo)).sum()  # Todo correcto
FP = ((y_real_ataque == 0) & (y_pred_ataque == 1)).sum()  # Falsos positivos en ataque
FN = ((y_real_ataque == 1) & (y_pred_ataque == 0)).sum()  # Falsos negativos en ataque
TN = ((y_real_ataque == 0) & (y_pred_ataque == 0)).sum()  # Verdaderos negativos

# Casos donde el ataque es correcto, pero hay errores en categoría o tipo
error_categoria = ((y_real_ataque == 1) & (y_pred_ataque == 1) & (y_real_categoria != y_pred_categoria)).sum()
error_tipo = ((y_real_ataque == 1) & (y_pred_ataque == 1) & (y_real_categoria == y_pred_categoria) & (y_real_tipo != y_pred_tipo)).sum()

# Imprimir resultados
print("Matriz de confusión extendida:")
print(f"TP (Ataque, Categoría y Tipo correctos): {TP}")
print(f"FP (Falsos Positivos - Ataque detectado incorrectamente): {FP}")
print(f"FN (Falsos Negativos - Ataque no detectado): {FN}")
print(f"TN (Verdaderos Negativos - No ataque correctamente identificado): {TN}")
print(f"Errores en Categoría cuando el ataque es correcto: {error_categoria}")
print(f"Errores en Tipo cuando el ataque y la categoría son correctos: {error_tipo}")

# Ajustar TP y FP considerando errores en categoría y tipo
TP_corrected = TP - error_categoria - error_tipo  # Restamos los errores dentro de TP
FP_corrected = FP + error_categoria + error_tipo  # Sumamos los errores dentro de FP

# Recalcular métricas
accuracy = (TP_corrected + TN) / (TP_corrected + FP_corrected + FN + TN)
precision = TP_corrected / (TP_corrected + FP_corrected)
recall = TP_corrected / (TP_corrected + FN)
f1 = 2 * (precision * recall) / (precision + recall)
specificity = TN / (TN + FP_corrected)

# Imprimir nuevas métricas
print(f"Accuracy corregido: {accuracy:.4f}")
print(f"Precision corregida: {precision:.4f}")
print(f"Recall corregido: {recall:.4f}")
print(f"F1-Score corregido: {f1:.4f}")
print(f"Specificity corregida: {specificity:.4f}")

