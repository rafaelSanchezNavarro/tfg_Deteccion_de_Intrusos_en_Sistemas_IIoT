import os

import joblib

from scripts.test import test

# model_path = os.path.join("modelos", f"DecisionTreeClassifier_0.9935", f"DecisionTreeClassifier_0.9935.pkl")
# model_test = joblib.load(model_path)
# print(model_test)
# test.main(model_test)


# import os

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
# from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
# from scripts.preprocesamiento.conversion import convert_numeric_columns


# import os
# import pandas as pd
# import joblib

# def cargar_datos():
#     """Carga todos los archivos procesados, el preprocesamiento y los devuelve como un diccionario."""
#     carpeta = r"datos/preprocesados"
#     datos = {}

#     # Cargar X_test
#     path_X_test = os.path.join(carpeta, "X_test_sin_pre.csv")
#     try:
#         datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
#         print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
#     except FileNotFoundError:
#         print(f"❌ Error: No se encontró {path_X_test}.")
#         return None

#     # Cargar y_test_class3
#     path_y_test_class3 = os.path.join(carpeta, "y_test_class3_sin_pre.csv")
#     try:
#         datos["y_test_class3"] = pd.read_csv(path_y_test_class3, low_memory=False)
#         print(f"✅ y_test_class3 cargado: {datos['y_test_class3'].shape[0]} filas, {datos['y_test_class3'].shape[1]} columnas.")
#     except FileNotFoundError:
#         print(f"❌ Error: No se encontró {path_y_test_class3}.")
#         return None

#     # Cargar componentes de preprocesamiento
#     try:
#         path = os.path.join("modelos", "DecisionTreeClassifier_0.9935")
#         datos["imputador_cat"] = joblib.load(os.path.join(path, "imputador_cat.pkl"))
#         datos["imputador_num"] = joblib.load(os.path.join(path, "imputador_num.pkl"))
#         datos["normalizacion"] = joblib.load(os.path.join(path, "normalizacion.pkl"))
#         datos["discretizador"] = joblib.load(os.path.join(path, "discretizador.pkl"))
#         datos["decodificador"] = joblib.load(os.path.join(path, "decodificador.pkl"))
#         datos["caracteristicas"] = joblib.load(os.path.join(path, "caracteristicas.pkl"))
#         print(f"✅ Preprocesamiento cargado: {path}")
#     except FileNotFoundError:
#         print("❌ Error: No se encontró el preprocesamiento.")
#         return None

#     return datos


# def main():  
#     print("🚀 Iniciando test...")
    
#     # Cargar todos los archivos de datos procesados
#     datos = cargar_datos()

#     # Asignar los DataFrames a variables individuales
#     X_test = datos["X_test"]
#     y_test_class3 = datos["y_test_class3"]
#     y_test_class3 = y_test_class3['class3'].to_numpy().flatten()
#     print(y_test_class3)

# if __name__ == "__main__":
#     main()