# import matplotlib.pyplot as plt
# import numpy as np

# accuracy_old = [0.9999, 1.0000, 1.0000, 1.0000, 1.0000]
# precision_old = [0.9983, 1.0000, 1.0000, 1.0000, 1.0000]
# recall_old = [0.9999, 1.0000, 1.0000, 1.0000, 1.0000]
# f1_old = [0.9991, 1.0000, 1.0000, 1.0000, 1.0000]






# accuracy_new = [0.9997, 1.0000, 1.0000, 1.0000, 1.0000]
# precision_new = [0.9998, 1.0000, 1.0000, 1.0000, 1.0000]
# recall_new = [0.9998, 1.0000, 1.0000, 1.0000, 1.0000]
# f1_new = [0.9998, 1.0000, 1.0000, 1.0000, 1.0000]






# # Calcular la media de cada métrica
# mean_old = [np.mean(accuracy_old), np.mean(precision_old), np.mean(recall_old), np.mean(f1_old)]
# mean_new = [np.mean(accuracy_new), np.mean(precision_new), np.mean(recall_new), np.mean(f1_new)]

# # Métricas
# metricas = ['Tasa de acierto', 'Precisión', 'Sensibilidad', 'F1-score']

# # Graficar
# x = np.arange(len(metricas))
# width = 0.3

# plt.figure(figsize=(8, 5))
# plt.bar(x - width/2, mean_old, width, label="Modelo multiclase", color="#F8CECC", edgecolor="grey")
# plt.bar(x + width/2, mean_new, width, label="Modelo multiclase balanceado", color="#D5E8D4", edgecolor="grey")

# # Personalizar
# plt.xticks(x, metricas, fontsize=12)
# plt.ylabel("Valor Promedio", fontsize=12)
# plt.title("Comparación de métricas promedio entre modelos multiclase para la clasificacion del tipo", fontsize=10)
# plt.ylim(0.999, 1)  # Establecer el rango del eje Y


# # Mejorar diseño
# plt.gca().spines["top"].set_visible(False)
# plt.gca().spines["right"].set_visible(False)
# plt.legend(loc='lower left')
# # Mostrar gráfico
# plt.show()


# from matplotlib import pyplot as plt
# import numpy as np

# metrics = ['Tasa de acierto', 'Precisión', 'Sensibilidad', 'F1-score']

# # accuracy1 = [0.9998, 0.9999, 1.0000, 1.0000, 1.0000]
# # precision1 = [0.9999, 1.0000, 1.0000, 1.0000, 1.0000]
# # recall1 = [0.9998, 0.9999, 1.0000, 1.0000, 1.0000]
# # f11 = [0.9998, 0.9999, 1.0000, 1.0000, 1.0000]

# # accuracy3 = [0.9999, 1.0000, 1.0000, 1.0000, 1.0000]
# # precision3 = [0.9983, 1.0000, 1.0000, 1.0000, 1.0000]
# # recall3 = [0.9999, 1.0000, 1.0000, 1.0000, 1.0000]
# # f13 = [0.9991, 1.0000, 1.0000, 1.0000, 1.0000]
# # # Calcular la media de cada métrica
# # mean1 = [np.mean(accuracy1), np.mean(precision1), np.mean(recall1), np.mean(f11)]
# # mean2 = [np.mean(accuracy3), np.mean(precision3), np.mean(recall3), np.mean(f13)]

# arch1 = [0.9938, 0.9710, 0.9735, 0.9722]
# arch2 = [0.9940, 0.9686, 0.9699, 0.9692]


# arch3 = [0.9937, 0.9490, 0.9347, 0.9400]

# # arch4 = [np.mean(accuracy1), np.mean(precision1), np.mean(recall1), np.mean(f11)]
# # arch5 = [np.mean(accuracy3), np.mean(precision3), np.mean(recall3), np.mean(f13)]

# # Colores asignados
# color_arch1 = '#D5E8D4'
# color_arch2 = '#F8CECC'
# color_arch3 = '#DAE8FC'  


# # # # Crear gráfico
# plt.figure(figsize=(10, 6))
# # plt.plot(metrics, arch4, marker='o', label='Arquitectura 1 balanceada', color='grey', linewidth=3, markeredgecolor='grey', linestyle='--', markerfacecolor='white') 
# plt.plot(metrics, arch1, marker='o', label='Modelo multiclase', color=color_arch2, linewidth=3, markeredgecolor='grey')
# plt.plot(metrics, arch2, marker='o', label='Modelo multiclase balanceado', color=color_arch1, linewidth=3, markeredgecolor='grey')
# # plt.plot(metrics, arch5, marker='o', label='Arquitectura 3 balanceada', color='grey', linewidth=3, markeredgecolor='grey', linestyle='--', markerfacecolor='white') 
# # plt.plot(metrics, arch3, marker='o', label='Arquitectura 3 balanceada (predicciones heredadas)', color=color_arch3, linewidth=3, markeredgecolor='grey')



# plt.title('Comparación entre el modelo multiclase y modelo multiclase balanceado')
# plt.ylabel('Valor')
# plt.ylim(0.96, 1)  # Establecer el rango del eje Y
# plt.grid(False)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.lines import Line2D  # Para crear leyendas personalizadas

# # Datos originales
# x = np.arange(9)
# y = np.array([0, 25, 8, 50, 5, 9, 20, 10, 17])

# # Normalización de X e Y
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# x_norm = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
# y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# # Colores
# color_arch1 = '#D5E8D4'  # Normalizado
# color_arch2 = '#F8CECC'  # Original

# # Crear figura principal
# fig, ax = plt.subplots(figsize=(8, 6))

# # Datos originales
# ax.scatter(x, y, marker='^', color=color_arch2, label='Original', edgecolor='grey')

# # Crear eje insertado para la normalización
# ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
# ax_inset.scatter(x_norm, y_norm, edgecolor='grey', facecolor=color_arch1, marker='o', label='Normalizado')
# ax_inset.tick_params(labelsize=8)

# legend_elements = [
#     Line2D([0], [0], marker='^', color='w', label='Original',
#            markerfacecolor=color_arch2, markeredgecolor='grey', markersize=10),
#     Line2D([0], [0], marker='o', color='w', label='Normalizado',
#            markerfacecolor=color_arch1, markeredgecolor='grey', markersize=10)
# ]

# # Ajustes visuales
# ax.set_title('Datos originales vs datos normalizados')
# ax.legend(handles=legend_elements, loc='upper left')
# ax.grid(False)
# ax_inset.grid(False)

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Cargar dataset iris desde seaborn
# iris = sns.load_dataset('iris')

# # Seleccionar columnas de interés
# x = iris['sepal_length']
# y = iris['sepal_width']

# color_arch1 = '#D5E8D4'  # Normalizado
# color_arch2 = '#F8CECC'  # Original

# # Detectar outliers usando el método IQR en ambas variables
# def detectar_outliers(series):
#     Q1 = series.quantile(0.30)
#     Q3 = series.quantile(0.70)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     return (series < lower) | (series > upper)

# # Detectar outliers combinados
# outliers_x = detectar_outliers(x)
# outliers_y = detectar_outliers(y)
# outliers = outliers_x | outliers_y

# # Crear figura
# plt.figure(figsize=(8, 6))

# # Puntos normales
# plt.scatter(x[~outliers], y[~outliers], color=color_arch1, label='Normal', edgecolor='grey', marker='o')

# # Outliers
# plt.scatter(x[outliers], y[outliers], color=color_arch2, label='Outlier', edgecolor='grey', marker='^')

# # Estética del gráfico
# plt.xlabel('Sepal Width (cm)')
# plt.ylabel('Sepal Length (cm)')
# plt.title('Detección de valores atípicos - Conjunto de datos Iris')
# plt.legend()

# plt.tight_layout()
# plt.show()



# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.lines import Line2D

# # Datos originales
# categorias = [
#     "RDOS", "Reconnaissance", "Weaponization", "Lateral _movement",
#     "Exfiltration", "Tampering", "C&C", "Exploitation", "crypto-ransomware"
# ]
# valores = [98942, 89471, 47043, 22034, 15421, 3605, 1973, 788, 315]
# total = sum(valores)

# # Separar en categorías principales y menores (< 15,000)
# categorias_principales = []
# valores_principales = []
# categorias_menores = []
# valores_menores = []

# for cat, val in zip(categorias, valores):
#     if val < 15000:
#         categorias_menores.append(cat)
#         valores_menores.append(val)
#     else:
#         categorias_principales.append(cat)
#         valores_principales.append(val)

# # Colores personalizados
# color_arch1 = '#D5E8D4'  # para menores
# color_arch2 = '#F8CECC'  # para principales
# colores_principales = [color_arch2] * len(categorias_principales)
# colores_menores = [color_arch1] * len(categorias_menores)

# # Crear figura principal
# fig, ax = plt.subplots(figsize=(10, 6))

# # Gráfico de barras principales
# bars = ax.bar(categorias_principales, valores_principales, color=colores_principales, edgecolor='grey')
# ax.set_title("Distribución de categorías de ataques")
# ax.set_ylabel("Número de instancias")
# ax.set_xticks(range(len(categorias_principales)))
# ax.set_xticklabels(categorias_principales, rotation=45, ha='right')

# # Añadir porcentaje en el centro de cada barra principal
# for bar, valor in zip(bars, valores_principales):
#     height = bar.get_height()
#     porcentaje = f"{(valor / total) * 100:.2f}%"
#     ax.text(bar.get_x() + bar.get_width() / 2, height / 2, porcentaje,
#             ha='center', va='center', fontsize=9, color='black')

# # Subgráfico insertado para menores
# ax_inset = inset_axes(ax, width="45%", height="45%", loc='upper right')
# bars_inset = ax_inset.bar(categorias_menores, valores_menores, color=colores_menores, edgecolor='grey')
# ax_inset.set_xticks(range(len(categorias_menores)))
# ax_inset.set_xticklabels(categorias_menores, rotation=45, ha='right')
# ax_inset.tick_params(labelsize=8)

# # Añadir porcentaje en el centro de cada barra menor
# for bar, valor in zip(bars_inset, valores_menores):
#     height = bar.get_height()
#     porcentaje = f"{(valor / total) * 100:.2f}%"
#     ax_inset.text(bar.get_x() + bar.get_width() / 2, height / 2, porcentaje,
#                   ha='center', va='center', fontsize=8, color='black')

# # Leyenda personalizada
# legend_elements = [
#     Line2D([0], [0], marker='s', color='w', label='Categorías con mas de 15,000 instancias',
#            markerfacecolor=color_arch2, markeredgecolor='grey', markersize=10),
#     Line2D([0], [0], marker='s', color='w', label='Categorías con menos de 15,000 instancias',
#            markerfacecolor=color_arch1, markeredgecolor='grey', markersize=10)
# ]
# ax.legend(handles=legend_elements, loc='lower left')

# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.lines import Line2D

# # Datos actualizados
# categorias = [
#     "RDOS", "Scanning_vulnerability", "Generic_scanning", "BruteForce",
#     "MQTT_cloud_broker_subscription", "Discovering_resources", "Exfiltration",
#     "insider_malcious", "Modbus_register_reading", "False_data_injection", "C&C",
#     "Dictionary", "TCP Relay", "fuzzing", "Reverse_shell",
#     "crypto-ransomware", "MitM", "Fake_notification"
# ]

# valores = [
#     98942,   # RDOS
#     37070,   # Scanning_vulnerability
#     35185,   # Generic_scanning
#     33046,   # BruteForce
#     16456,   # MQTT_cloud_broker_subscription
#     16284,   # Discovering_resources
#     15421,   # Exfiltration
#     12184,   # insider_malcious
#      4130,   # Modbus_register_reading
#      3586,   # False_data_injection
#      1973,   # C&C
#      1813,   # Dictionary
#      1448,   # TCP Relay
#       932,   # fuzzing
#       701,   # Reverse_shell
#       315,   # crypto-ransomware
#        87,   # MitM
#        19    # Fake_notification
# ]

# total = sum(valores)

# # Separar en principales y menores (< 20,000)
# categorias_principales = []
# valores_principales = []
# categorias_menores = []
# valores_menores = []

# for cat, val in zip(categorias, valores):
#     if val < 3000:
#         categorias_menores.append(cat)
#         valores_menores.append(val)
#     else:
#         categorias_principales.append(cat)
#         valores_principales.append(val)

# # Colores personalizados
# color_arch1 = '#D5E8D4'  # para menores
# color_arch2 = '#F8CECC'  # para principales
# colores_principales = [color_arch2] * len(categorias_principales)
# colores_menores = [color_arch1] * len(categorias_menores)

# # Crear figura principal
# fig, ax = plt.subplots(figsize=(12, 7))

# # Gráfico de barras principales
# bars = ax.bar(categorias_principales, valores_principales, color=colores_principales, edgecolor='grey')
# ax.set_title("Distribución de tipos de ataque")
# ax.set_ylabel("Número de instancias")
# ax.set_xticks(range(len(categorias_principales)))
# ax.set_xticklabels(categorias_principales, rotation=45, ha='right')

# # Añadir porcentaje en el centro de cada barra principal
# for bar, valor in zip(bars, valores_principales):
#     height = bar.get_height()
#     porcentaje = f"{(valor / total) * 100:.2f}%"
#     ax.text(bar.get_x() + bar.get_width() / 2, height / 2, porcentaje,
#             ha='center', va='center', fontsize=9, color='black')

# # Subgráfico insertado para menores
# ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')
# bars_inset = ax_inset.bar(categorias_menores, valores_menores, color=colores_menores, edgecolor='grey')
# ax_inset.set_xticks(range(len(categorias_menores)))
# ax_inset.set_xticklabels(categorias_menores, rotation=45, ha='right')
# ax_inset.tick_params(labelsize=8)

# # Añadir porcentaje en el centro de cada barra menor
# for bar, valor in zip(bars_inset, valores_menores):
#     height = bar.get_height()
#     porcentaje = f"{(valor / total) * 100:.2f}%"
#     ax_inset.text(bar.get_x() + bar.get_width() / 2, height / 2, porcentaje,
#                   ha='center', va='center', fontsize=8, color='black')

# # Leyenda personalizada
# legend_elements = [
#     Line2D([0], [0], marker='s', color='w', label='Categorías con mas de 3,000 instancias',
#            markerfacecolor=color_arch2, markeredgecolor='grey', markersize=10),
#     Line2D([0], [0], marker='s', color='w', label='Categorías con menos de 3,000 instancias',
#            markerfacecolor=color_arch1, markeredgecolor='grey', markersize=10)
# ]
# ax.legend(handles=legend_elements, loc='lower left')

# plt.tight_layout()
# plt.show()




