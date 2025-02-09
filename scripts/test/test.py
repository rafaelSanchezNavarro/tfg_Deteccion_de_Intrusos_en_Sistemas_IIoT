import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
from scripts.preprocesamiento.conversion import convert_numeric_columns


import os
import pandas as pd
import joblib

def cargar_datos(pre_path):
    """Carga todos los archivos procesados, el preprocesamiento y los devuelve como un diccionario."""
    carpeta = r"datos/preprocesados"
    datos = {}

    # Cargar X_test
    path_X_test = os.path.join(carpeta, "X_test_sin_pre.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None

    # Cargar y_test_class3
    path_y_test_class3 = os.path.join(carpeta, "y_test_class3_sin_pre.csv")
    try:
        datos["y_test_class3"] = pd.read_csv(path_y_test_class3, low_memory=False)
        print(f"✅ y_test_class3 cargado: {datos['y_test_class3'].shape[0]} filas, {datos['y_test_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class3}.")
        return None

    # Cargar componentes de preprocesamiento
    try:
        path = os.path.join("modelos", pre_path)
        datos["imputador_cat"] = joblib.load(os.path.join(path, "imputador_cat.pkl"))
        datos["imputador_num"] = joblib.load(os.path.join(path, "imputador_num.pkl"))
        datos["normalizacion"] = joblib.load(os.path.join(path, "normalizacion.pkl"))
        datos["discretizador"] = joblib.load(os.path.join(path, "discretizador.pkl"))
        datos["decodificador"] = joblib.load(os.path.join(path, "decodificador.pkl"))
        datos["caracteristicas"] = joblib.load(os.path.join(path, "caracteristicas.pkl"))
        print(f"✅ Preprocesamiento cargado: {path}")
    except FileNotFoundError:
        print("❌ Error: No se encontró el preprocesamiento.")
        return None

    return datos

def preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador, caracteristicas):
    X_test = convert_numeric_columns(X_test)
    X_test = replace_common_values(X_test)
    X_test = fix_mayus(X_test)

    
    # Identificar columnas categóricas, numéricas y booleanas
    categorical_cols = X_test.select_dtypes(include=['object']).columns
    boolean_cols = X_test.select_dtypes(include=['bool']).columns
    if boolean_cols.any():  # Si hay columnas booleanas
        X_test[boolean_cols] = X_test[boolean_cols].astype(int)
    numerical_cols = X_test.select_dtypes(include=['float64', 'int64']).columns
    
    ##############################################################################

    X_test[categorical_cols] = imputador_cat.fit_transform(X_test[categorical_cols])
    
    X_test[numerical_cols] = imputador_num.fit_transform(X_test[numerical_cols])

    ##############################################################################
    
    X_test_scaled = normalizacion.fit_transform(X_test[numerical_cols])

    # Convertir las matrices escaladas a DataFrames
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=[f"{col}_scaled" for col in numerical_cols], index=X_test.index)

    ##############################################################################
    
    X_test_discrete = discretizador.fit_transform(X_test[numerical_cols])

    # Convertir las matrices discretizadas a DataFrames
    X_test_discretized_df = pd.DataFrame(X_test_discrete, columns=[f"{col}_discrete" for col in numerical_cols], index=X_test.index)
    
    ##############################################################################
    
    processed_numeric_test = pd.concat([X_test_scaled_df, X_test_discretized_df], axis=1)
    
    ##############################################################################
    
    X_test_encoded = decodificador.fit_transform(X_test[categorical_cols])

    # Obtener los nombres de las nuevas columnas codificadas
    encoded_cols = decodificador.get_feature_names_out(categorical_cols)

    # Convertir las matrices codificadas a DataFrames
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
    
    ##############################################################################
    
    # Combinar con las características categóricas codificadas
    X_test_processed = pd.concat([processed_numeric_test, X_test_encoded_df], axis=1)
    
    # Opcional: Reordenar las columnas si es necesario
    X_test_processed = X_test_processed.reindex(sorted(X_test_processed.columns), axis=1)
    
    caracteristicas = caracteristicas.tolist()
    X_test_processed = X_test_processed[caracteristicas]
    
    return X_test_processed 
    
def main(model, path):  
    print("🚀 Iniciando test...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos(path)

    # Asignar los DataFrames a variables individuales
    X_test = datos["X_test"]
    y_test_class3 = datos["y_test_class3"]
    imputador_cat = datos["imputador_cat"]
    imputador_num = datos["imputador_num"]
    normalizacion = datos["normalizacion"]
    discretizador = datos["discretizador"]
    decodificador = datos["decodificador"]
    caracteristicas = datos["caracteristicas"]
    
    # Preprocesar los datos de test
    X_test_processed = preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador, caracteristicas)
    print(f"✅ Preprocesamiento de test finalizado: {X_test_processed.shape[0]} filas, {X_test_processed.shape[1]} columnas.")
    
    
    # Realizar predicciones
    y_pred_class3 = model.predict(X_test_processed)

    # Si necesitas probabilidades
    probabilities = model.predict_proba(X_test_processed)
    
    accuracy = accuracy_score(y_test_class3, y_pred_class3)
    print(f"Accuracy (Test): {accuracy:.4f}")
    
    precision = precision_score(y_test_class3, y_pred_class3)
    print(f"Precision (Test): {precision:.4f}")
    
    recall = recall_score(y_test_class3, y_pred_class3)
    print(f"Recall (Test): {recall:.4f}")
    
    f1 = f1_score(y_test_class3, y_pred_class3)
    print(f"F1 (Test): {f1:.4f}")
    
    # return accuracy, precision, recall, f1
