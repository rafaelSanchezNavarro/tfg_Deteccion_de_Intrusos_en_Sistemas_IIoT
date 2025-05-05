import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from scripts.anomalias import anomalias
from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
from scripts.preprocesamiento.conversion import delete_ip_port, fix_dtype
from scripts.test.evaluacion import evaluacion_tipo

def cargar_datos(pre_path):

    carpeta = r"datos/preprocesados"
    datos = {}

    path_X_test = os.path.join(carpeta, "X_test.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None

    path_y_test_class3 = os.path.join(carpeta, "y_test_class3.csv")
    try:
        datos["y_test_class3"] = pd.read_csv(path_y_test_class3, low_memory=False)
        print(f"✅ y_test_class3 cargado: {datos['y_test_class3'].shape[0]} filas, {datos['y_test_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class3}.")
        return None
    
    path_y_test_class2 = os.path.join(carpeta, "y_test_class2.csv")
    try:
        datos["y_test_class2"] = pd.read_csv(path_y_test_class2, low_memory=False)
        print(f"✅ y_test_class2 cargado: {datos['y_test_class2'].shape[0]} filas, {datos['y_test_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class2}.")
        return None
    
    path_y_test_class1 = os.path.join(carpeta, "y_test_class1.csv")
    try:
        datos["y_test_class1"] = pd.read_csv(path_y_test_class1, low_memory=False)
        print(f"✅ y_test_class1 cargado: {datos['y_test_class1'].shape[0]} filas, {datos['y_test_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class1}.")
        return None

    try:
        path = os.path.join("modelos", pre_path)
        datos["imputador_cat"] = joblib.load(os.path.join(path, "imputador_cat.pkl"))
        datos["imputador_num"] = joblib.load(os.path.join(path, "imputador_num.pkl"))
        datos["normalizacion"] = joblib.load(os.path.join(path, "normalizacion.pkl"))
        if os.path.exists(os.path.join(path, "discretizador.pkl")):
            datos["discretizador"] = joblib.load(os.path.join(path, "discretizador.pkl"))
        datos["decodificador"] = joblib.load(os.path.join(path, "decodificador.pkl"))
        datos["caracteritisticas_seleccionadas"] = joblib.load(os.path.join(path, "caracteritisticas_seleccionadas.pkl"))
        datos["caracteritisticas_procesadas"] = joblib.load(os.path.join(path, "caracteritisticas_procesadas.pkl"))
        print(f"✅ Preprocesamiento cargado: {path}")
    except FileNotFoundError:
        print("❌ Error: No se encontró el preprocesamiento.")
        return None

    return datos

def guardar_conf(pre_path, metrics_class2, metrics_class1, cm_df, class_df, cm_dfs, class_dfs):
    
    resumen = crear_resumen(metrics_class2, metrics_class1, cm_df, class_df, cm_dfs, class_dfs)
    path_resumen = os.path.join(f"modelos/{pre_path}", "resumen_test.txt")
    with open(path_resumen, "w", encoding="utf-8") as f:
        f.write(resumen)

def guardar_datos(X_test_processed, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    X_test_processed.to_csv(os.path.join(output_dir, "X_test_preprocesado.csv"), index=False)

def crear_resumen(metrics_class2, metrics_class1, cm_df, class_df, cm_dfs, class_dfs):
    
    def formatear_metricas(nombre, metricas, df_cm=None, df_class=None):
        resultado = (
            f"{nombre}\n"
            f"  - Accuracy: {metricas[0]:.4f}\n"
            f"  - Precision: {metricas[1]:.4f}\n"
            f"  - Recall: {metricas[2]:.4f}\n"
            f"  - F1: {metricas[3]:.4f}\n"
        )
        
        if df_cm is not None:
            resultado += f"  - Confusion Matrix:\n{df_cm}\n"
        
        if df_class is not None:
            resultado += f"  - Classification Report:\n{df_class}\n"
        
        return resultado

    resumen = "Resumen de métricas\n\n"
    resumen += formatear_metricas("\nClasificación Multiclase (Categoría)", metrics_class2, cm_df, class_df)
    resumen += "\nClasificación Multiclase (Tipo)\n"
    
    for nombre, metricas in metrics_class1.items():
        resumen += formatear_metricas(nombre, metricas, cm_dfs[nombre], class_dfs[nombre])

    return resumen

def preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador):
    
    categorical_cols = X_test.select_dtypes(include=['object']).columns
    boolean_cols = X_test.select_dtypes(include=['bool']).columns
    if boolean_cols.any():  
        X_test[boolean_cols] = X_test[boolean_cols].astype(float) # TAL VEZ INNCESESARIO
    numerical_cols = X_test.select_dtypes(include=['float64', 'int64']).columns
        
    ##############################################################################
    
    X_test[categorical_cols] = imputador_cat.transform(X_test[categorical_cols])
    X_test[numerical_cols] = imputador_num.transform(X_test[numerical_cols])
    
    ##############################################################################
    
    X_test_scaled = normalizacion.transform(X_test[numerical_cols])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=[f"{col}_scaled" for col in numerical_cols], index=X_test.index)

    ##############################################################################
    
    if discretizador is not None:
        X_test_discrete = discretizador.transform(X_test[numerical_cols])
        X_test_discretized_df = pd.DataFrame(X_test_discrete, columns=[f"{col}_discrete" for col in numerical_cols], index=X_test.index)
    
    ##############################################################################
    
    if discretizador is not None:
        processed_numeric_test = pd.concat([X_test_scaled_df, X_test_discretized_df], axis=1)
    else:
        processed_numeric_test = X_test_scaled_df
    
    ##############################################################################
    
    X_test_encoded = decodificador.transform(X_test[categorical_cols])
    encoded_cols = decodificador.get_feature_names_out(categorical_cols)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
    
    ##############################################################################
    
    X_test_processed = pd.concat([processed_numeric_test, X_test_encoded_df], axis=1)
    X_test_processed = X_test_processed.reindex(sorted(X_test_processed.columns), axis=1)
    
    return X_test_processed 
   
def graficar_roc(y_test_class3, y_pred_class3):
    
    fpr, tpr, thresholds = roc_curve(y_test_class3, y_pred_class3)
    roc_auc = roc_auc_score(y_test_class3, y_pred_class3)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
 
def clasificacion_multiclase_categoria(y_test_class2, X_test_processed, model_class2):

        print("🔮 Clasificación multiclase (Categoría)...")

        y_test_class2 = y_test_class2.values.ravel()
        y_pred_class2 = model_class2.predict(X_test_processed)

        accuracy = accuracy_score(y_test_class2, y_pred_class2)
        print(f'📈 Accuracy (Test): {accuracy:.4f}')
        precision = precision_score(y_test_class2, y_pred_class2, average='macro', zero_division=0)
        print(f'📈 Precision (Test): {precision:.4f}')
        recall = recall_score(y_test_class2, y_pred_class2, average='macro')
        print(f'📈 Recall (Test): {recall:.4f}')
        f1 = f1_score(y_test_class2, y_pred_class2, average='macro')
        print(f'📈 F1 (Test): {f1:.4f}')
        
        labels = sorted(set(y_test_class2))
        label_names = [str(label) for label in labels] 
        cm = confusion_matrix(y_test_class2, y_pred_class2)
        cm_df = pd.DataFrame(cm, index=[f'{label}' for label in label_names], columns=[f'{label}' for label in label_names])
        print(cm_df)
        class_df = classification_report(y_test_class2, y_pred_class2, target_names=label_names, digits=10)
        
        predicciones_df = pd.DataFrame({'Predicciones_Class2': y_pred_class2})    
        ruta_directorio = 'predicciones'
        os.makedirs(ruta_directorio, exist_ok=True)
        predicciones_df.to_csv(os.path.join(ruta_directorio, 'arq3.csv'), index=False)
        
        metrics = [accuracy, precision, recall, f1]
        return metrics, y_pred_class2, cm_df, class_df
 
def clasificacion_multiclase_tipo(y_pred_class2, y_test_class1, X_test_processed, models_class1):

    print("🔮 Clasificación multiclase (Tipo)...")
    
    metrics = {}
    cm_dfs = {}
    class_dfs = {}  
    
    indices_anomalia_test = np.where(y_pred_class2 != "Normal")[0]
    X_test_processed = X_test_processed.iloc[indices_anomalia_test]
    y_pred_class2 = y_pred_class2[indices_anomalia_test]
    y_test_class1 = y_test_class1.iloc[indices_anomalia_test].values.ravel()
    

    y_pred_class1_total = ["Normal"] * len(indices_anomalia_test)  # Inicializa con "Normal"
    
    # Cargar el CSV existente
    ruta_directorio = 'predicciones'
    predicciones_df = pd.read_csv(os.path.join(ruta_directorio, 'arq3.csv'))

    categorias_multiples_tipos = [key for key, value in anomalias.items() if len(value) > 1]
    
    for categoria in categorias_multiples_tipos:
        
        print(f"🔮 Clasificación multiclase (Tipo) para {categoria}...")
        indices_anomalia_prediccion_categoria = np.where(y_pred_class2 == categoria)[0]
        X_test_categoria = X_test_processed.iloc[indices_anomalia_prediccion_categoria]
        y_test_class1_categoria = y_test_class1[indices_anomalia_prediccion_categoria]
        
        model = models_class1[categoria]

        y_pred_class1_categoria = model.predict(X_test_categoria)
        
        # Asignar las predicciones a la lista total usando el índice correcto
        for idx, pred in zip(indices_anomalia_prediccion_categoria, y_pred_class1_categoria):
            y_pred_class1_total[idx] = pred
        
        indices_anomalias_reales = np.where(np.isin(y_test_class1_categoria, anomalias.get(categoria, [])))[0]
        y_test_class1_real = y_test_class1_categoria[indices_anomalias_reales]
        y_pred_class1_real = y_pred_class1_categoria[indices_anomalias_reales]

        accuracy = accuracy_score(y_test_class1_real, y_pred_class1_real)
        print(f'📈 Accuracy: {accuracy:.4f}')
        precision = precision_score(y_test_class1_real, y_pred_class1_real, average='macro', zero_division=0)
        print(f'📈 Precision: {precision:.4f}')
        recall = recall_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 Recall: {recall:.4f}')
        f1 = f1_score(y_test_class1_real, y_pred_class1_real, average='macro')
        print(f'📈 F1: {f1:.4f}')    
        
        labels = sorted(set(y_test_class1_real))
        label_names = [str(label) for label in labels]  
        cm = confusion_matrix(y_test_class1_real, y_pred_class1_real)
        cm_df = pd.DataFrame(cm, index=[f'{label}' for label in label_names], columns=[f'{label}' for label in label_names])
        print(cm_df)
        class_df = classification_report(y_test_class1_real, y_pred_class1_real, target_names=label_names, digits=10)
        print(class_df)
        cm_dfs[categoria] = cm_df
        class_dfs[categoria] = class_df
        
        metrics[categoria] = [accuracy, precision, recall, f1]
                    
        # Actualizar el DataFrame con las predicciones en cada iteración
        predicciones_df.loc[indices_anomalia_test, 'Predicciones_Class1'] = y_pred_class1_total 
        
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "RDOS"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "RDOS"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Exfiltration"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Exfiltration"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "C&C"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "C&C"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "crypto-ransomware"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "crypto-ransomware"
    
    normal_indices = predicciones_df[predicciones_df['Predicciones_Class2'] == "Normal"].index
    predicciones_df.loc[normal_indices, 'Predicciones_Class1'] = "Normal"

    # Guardar el DataFrame actualizado en el CSV
    os.makedirs(ruta_directorio, exist_ok=True)
    predicciones_df.to_csv(os.path.join(ruta_directorio, 'arq3.csv'), index=False)
            
    return metrics, predicciones_df, cm_dfs, class_dfs

def main(model_class2, path, models_class1):  
    print("🚀 Iniciando test...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos(path)

    # Asignar los DataFrames a variables individuales
    X_test = datos["X_test"]
    y_test_class3 = datos["y_test_class3"]
    y_test_class2 = datos["y_test_class2"]
    y_test_class1 = datos["y_test_class1"]
    imputador_cat = datos["imputador_cat"]
    imputador_num = datos["imputador_num"]
    normalizacion = datos["normalizacion"]
    if "discretizador" in datos:
        discretizador = datos["discretizador"]
    else:
        discretizador = None
    decodificador = datos["decodificador"]
    caracteritisticas_seleccionadas = datos["caracteritisticas_seleccionadas"]
    caracteritisticas_procesadas = datos["caracteritisticas_procesadas"]
    
    X_test = replace_common_values(X_test)
    X_test = fix_mayus(X_test)
    X_test = fix_dtype(X_test)
    X_test = delete_ip_port(X_test)
    
    y_test_class3 = y_test_class3.loc[X_test.index]
        
    X_test = X_test[caracteritisticas_seleccionadas] 
        
    X_test['Protocol'] = X_test['Protocol'].fillna("missing")
    
    # Preprocesar los datos de test
    X_test_processed = preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador)
    X_test_processed = X_test_processed[caracteritisticas_procesadas]
    print(f"✅ Preprocesamiento de test finalizado: {X_test_processed.shape[0]} filas, {X_test_processed.shape[1]} columnas.")

    output_dir = "datos/preprocesados"
    guardar_datos(X_test_processed,
                  output_dir)
    
    metrics_class2, y_pred_class2, cm_df, class_df = clasificacion_multiclase_categoria(y_test_class2, X_test_processed, model_class2)
    metrics_class1, y_pred_total, cm_dfs, class_dfs = clasificacion_multiclase_tipo(y_pred_class2, y_test_class1, X_test_processed, models_class1)
    
    guardar_conf(path, metrics_class2, metrics_class1, cm_df, class_df, cm_dfs, class_dfs)

    print("🎯 Test finalizado.\n")