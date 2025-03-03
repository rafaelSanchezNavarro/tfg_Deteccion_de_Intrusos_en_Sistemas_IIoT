# # import pandas as pd

# # # Cargar los datos desde el archivo CSV
# # data = pd.read_csv('predicciones.csv', header=None)

# # # Definir el diccionario que relaciona cada categoría con sus tipos permitidos
# # dicc = {
# #     "Reconnaissance": {
# #         "Generic_scanning", 
# #         "Scanning_vulnerability", 
# #         "Discovering_resources", 
# #         "fuzzing"
# #     },
# #     "Weaponization": {
# #         "BruteForce", 
# #         "Dictionary", 
# #         "insider_malcious"
# #     },
# #     "Exploitation": {
# #         "Reverse_shell", 
# #         "MitM"
# #     },
# #     "Lateral _movement": {
# #         "Modbus_register_reading", 
# #         "MQTT_cloud_broker_subscription", 
# #         "TCP Relay"
# #     },
# #     "C&C": {
# #         "C&C"
# #     },
# #     "Exfiltration": {
# #         "Exfiltration"
# #     },
# #     "Tampering": {
# #         "False_data_injection", 
# #         "Fake_notification"
# #     },
# #     "crypto-ransomware": {
# #         "crypto-ransomware"
# #     },
# #     "RDOS": {
# #         "RDOS"
# #     },
# #     "Normal": {
# #         "Normal"
# #     }
# # }


# # # Lista para almacenar las predicciones incorrectas
# # incorrectas = []

# # for index, row in data.iterrows():
# #     ataque = row[0]  # Categoria en ra1
# #     categoria = row[1]  # Predicción en ra2
# #     tipo = row[2]  # Predicción en ra3

# #     # Comprobar si la categoría es 0
# #     if ataque == 0:
# #         if categoria != 'Normal' or tipo != 'Normal':
# #             incorrectas.append((index, ataque, categoria, tipo))
    
# #     # Comprobar si la categoría es 1
# #     elif ataque == 1:
# #         if categoria in dicc:
# #             if tipo not in dicc[categoria]:
# #                 incorrectas.append((index, ataque, categoria, tipo))
# #         else:
# #             incorrectas.append((index, ataque, categoria, tipo))  # Categoría no válida
# #             print("Categoría no válida:", categoria)

# # print(len(incorrectas), "predicciones incorrectas:")
# # for error in incorrectas:
# #     print(f"Índice: {error[0]}, Ataque: {error[1]}, Categoría: {error[2]}, Tipo: {error[3]}")




import pandas as pd

# Cargar los dos CSV en DataFrames
csv1 = pd.read_csv('predicciones/real.csv', header=None)
csv2 = pd.read_csv('predicciones/predicciones_multiclase.csv', header=None)

# Asegurarse de que ambos DataFrames tengan el mismo número de filas
if len(csv1) != len(csv2):
    print("Los archivos tienen diferente número de filas. Ajustando al menor...")
    min_filas = min(len(csv1), len(csv2))
    csv1 = csv1.iloc[:min_filas, :3]
    csv2 = csv2.iloc[:min_filas, :3]
else:
    csv1 = csv1.iloc[:, :3]
    csv2 = csv2.iloc[:, :3]

# Comparar las filas completas (primeras 3 columnas) y encontrar las que no son iguales
diferencias = (csv1 != csv2).any(axis=1)

# Obtener los índices de las filas diferentes
indices_filas_diferentes = csv1.index[diferencias].tolist()

# Imprimir la cantidad de diferencias encontradas
print("Cantidad de filas diferentes:", len(indices_filas_diferentes))

# Imprimir los índices de las filas diferentes (los primeros 10 si hay más)
print("Índices de filas diferentes:", indices_filas_diferentes[:10] if len(indices_filas_diferentes) > 10 else indices_filas_diferentes)

