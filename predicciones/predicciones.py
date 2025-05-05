import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

###################################################################
# SUPERVISADO
##################################################################

anomalias = {
    "Reconnaissance": [
        "Generic_scanning",
        "Scanning_vulnerability",
        "fuzzing",
        "Discovering_resources"
    ],
    "Weaponization": [
        "BruteForce",
        "Dictionary",
        "insider_malcious"
    ],
    "Exploitation": [
        "Reverse_shell",
        "MitM"
    ],
    "Lateral _movement": [
        "MQTT_cloud_broker_subscription",
        "Modbus_register_reading",
        "TCP Relay"
    ],
    "C&C": [
        "C&C"
    ],
    "Exfiltration": [
        "Exfiltration"
    ],
    "Tampering": [
        "False_data_injection",
        "Fake_notification"
    ],
    "crypto-ransomware": [
        "crypto-ransomware"
    ],
    "RDOS": [
        "RDOS"
    ],
    "Normal": [
        "Normal"
    ]
} 

# Diccionario tipo → categoría
tipo_a_categoria = {}
for categoria, tipos in anomalias.items():
    for tipo in tipos:
        tipo_a_categoria[tipo] = categoria
        
# csv1 = pd.read_csv('predicciones/real.csv', header=None)
# csv2 = pd.read_csv('predicciones/pruebas arquitecturas/arq1b.csv', header=None)
# min_filas = min(len(csv1), len(csv2))
# csv1, csv2 = csv1.iloc[:min_filas, 2], csv2.iloc[:min_filas, 2]
# # csv2 = csv2.apply(lambda x: 0 if x == 'Normal' else 1)
# # csv2 = csv2.map(lambda x: tipo_a_categoria.get(x, x))  # <- ahora sí
# accuracy = accuracy_score(csv1, csv2)
# print(f'📈 Accuracy (test): {accuracy:.4f}')
# precision = precision_score(csv1, csv2, average='macro', zero_division=0)
# print(f'📈 Precision (test): {precision:.4f}')
# recall = recall_score(csv1, csv2, average='macro')
# print(f'📈 Recall (test): {recall:.4f}')
# f1 = f1_score(csv1, csv2, average='macro')
# print(f'📈 F1 (test): {f1:.4f}')




csv1 = pd.read_csv('predicciones/real.csv', header=None)
csv2 = pd.read_csv('predicciones/pruebas arq2b/GRU.csv', header=None)
min_filas = min(len(csv1), len(csv2))
csv1, csv2 = csv1.iloc[:min_filas, 1], csv2.iloc[:min_filas, 0]
# csv2 = csv2.apply(lambda x: 0 if x == 'Normal' else 1)
csv2 = csv2.map(lambda x: tipo_a_categoria.get(x, x))  # <- ahora sí
accuracy = accuracy_score(csv1, csv2)
print(f'📈 Accuracy (test): {accuracy:.4f}')
precision = precision_score(csv1, csv2, average='macro', zero_division=0)
print(f'📈 Precision (test): {precision:.4f}')
recall = recall_score(csv1, csv2, average='macro')
print(f'📈 Recall (test): {recall:.4f}')
f1 = f1_score(csv1, csv2, average='macro')
print(f'📈 F1 (test): {f1:.4f}')

labels = sorted(set(csv1))
label_names = [str(label) for label in labels]  
cm = confusion_matrix(csv1, csv2)
cm_df = pd.DataFrame(cm, index=[f'{label}' for label in label_names], columns=[f'{label}' for label in label_names])
print(cm_df)
class_df = classification_report(csv1, csv2, target_names=label_names, digits=4)
print(class_df)


###################################################################
# NO SUPERVISADO
###################################################################

# Cargar los dos CSV en DataFrames
# csv1 = pd.read_csv('predicciones/real.csv', header=None)
# csv2 = pd.read_csv('predicciones/arq2.csv', header=None)

# # Asegurar que ambos DataFrames tengan el mismo número de filas
# min_filas = min(len(csv1), len(csv2))
# y_real_ataque = csv1.iloc[:min_filas, 0]
# y_pred_ataque = csv2.iloc[:min_filas, 0]

# # Filtrar solo las filas donde el ataque (columna 0) es 1
# ataque_filas = (y_real_ataque == 1)

# # Definir casos en la matriz de confusión
# TP = ((y_real_ataque == 1) & (y_pred_ataque == 1)).sum()  # Ataque correctamente detectado
# FP = ((y_real_ataque == 0) & (y_pred_ataque == 1)).sum()  # Falsos positivos en ataque
# FN = ((y_real_ataque == 1) & (y_pred_ataque == 0)).sum()  # Falsos negativos en ataque
# TN = ((y_real_ataque == 0) & (y_pred_ataque == 0)).sum()  # Verdaderos negativos

# # Imprimir resultados
# print("Matriz de confusión extendida:")
# print(f"TP (Ataque correctamente detectado): {TP}")
# print(f"FP (Falsos Positivos - Ataque detectado incorrectamente): {FP}")
# print(f"FN (Falsos Negativos - Ataque no detectado): {FN}")
# print(f"TN (Verdaderos Negativos - No ataque correctamente identificado): {TN}")

# # Recalcular métricas
# accuracy = (TP + TN) / (TP + FP + FN + TN)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# f1 = 2 * (precision * recall) / (precision + recall)

# # Imprimir nuevas métricas
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")



# Matriz de confusión extendida:
# TP (Ataque correctamente detectado): 30096
# FP (Falsos Positivos - Ataque detectado incorrectamente): 31403
# FN (Falsos Negativos - Ataque no detectado): 0
# TN (Verdaderos Negativos - No ataque correctamente identificado): 1
# Accuracy: 0.4894
# Precision: 0.4894
# Recall: 1.0000
# F1-Score: 0.6572

# Matriz de confusión extendida:
# TP (Ataque correctamente detectado): 21064
# FP (Falsos Positivos - Ataque detectado incorrectamente): 22624
# FN (Falsos Negativos - Ataque no detectado): 38849
# TN (Verdaderos Negativos - No ataque correctamente identificado): 40589
# Accuracy: 0.5007
# Precision: 0.4821
# Recall: 0.3516
# F1-Score: 0.4066

# Matriz de confusión extendida:
# TP (Ataque correctamente detectado): 46989
# FP (Falsos Positivos - Ataque detectado incorrectamente): 49346
# FN (Falsos Negativos - Ataque no detectado): 12924
# TN (Verdaderos Negativos - No ataque correctamente identificado): 13867
# Accuracy: 0.4943
# Precision: 0.4878
# Recall: 0.7843
# F1-Score: 0.6015
