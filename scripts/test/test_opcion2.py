import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
from scripts.preprocesamiento.conversion import delete_ip_port, fix_dtype

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

def guardar_conf(pre_path, metrics_class1, cm_df_class1, class_df_class1):

    resumen = crear_resumen(metrics_class1, cm_df_class1, class_df_class1)
    path_resumen = os.path.join(f"modelos/{pre_path}", "resumen_test.txt")
    with open(path_resumen, "w", encoding="utf-8") as f:
        f.write(resumen)

def guardar_datos(X_test_processed, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    X_test_processed.to_csv(os.path.join(output_dir, "X_test_preprocesado.csv"), index=False)
    
def crear_resumen(metrics_class1, cm_df_class1, class_df_class1):
    
    def formatear_metricas(nombre, metricas, cm_df=None, class_df=None):
        resultado = (
            f"{nombre}\n"
            f"  - Accuracy: {metricas[0]:.4f}\n"
            f"  - Precision: {metricas[1]:.4f}\n"
            f"  - Recall: {metricas[2]:.4f}\n"
            f"  - F1: {metricas[3]:.4f}\n"
        )
        
        if cm_df is not None:
            resultado += f"  - Confusion Matrix:\n{cm_df}\n"
        
        if class_df is not None:
            resultado += f"  - Classification Report:\n{class_df}\n"
        
        return resultado

    resumen = "Resumen de métricas\n\n"
    resumen += formatear_metricas("\nClasificación Multiclase (Tipo)", metrics_class1, cm_df_class1, class_df_class1)
    
    return resumen

def preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador):

    categorical_cols = X_test.select_dtypes(include=['object']).columns
    boolean_cols = X_test.select_dtypes(include=['bool']).columns
    if boolean_cols.any():  # Si hay columnas booleanas
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

def clasificacion_multiclase_tipo(y_test_class1, X_test_processed, models_class1):

    print("🔮 Clasificación multiclase (Tipo)...")
    
    y_test_class1 = y_test_class1.values.ravel()
    y_pred_class1_cat = models_class1.predict(X_test_processed)

    accuracy = accuracy_score(y_test_class1, y_pred_class1_cat)
    print(f'📈 Accuracy: {accuracy:.4f}')
    precision = precision_score(y_test_class1, y_pred_class1_cat, average='macro', zero_division=0)
    print(f'📈 Precision: {precision:.4f}')
    recall = recall_score(y_test_class1, y_pred_class1_cat, average='macro')
    print(f'📈 Recall: {recall:.4f}')
    f1 = f1_score(y_test_class1, y_pred_class1_cat, average='macro')
    print(f'📈 F1: {f1:.4f}')
    
    
    labels = sorted(set(y_test_class1))
    label_names = [str(label) for label in labels]  
    cm = confusion_matrix(y_test_class1, y_pred_class1_cat)
    cm_df = pd.DataFrame(cm, index=[f'{label}' for label in label_names], columns=[f'{label}' for label in label_names])
    print(cm_df)
    class_df = classification_report(y_test_class1, y_pred_class1_cat, target_names=label_names, digits=4)
    
    predicciones_df = pd.DataFrame({'Predicciones_Class1': y_pred_class1_cat})    
    ruta_directorio = 'predicciones'
    os.makedirs(ruta_directorio, exist_ok=True)
    predicciones_df.to_csv(os.path.join(ruta_directorio, 'arq2.csv'), index=False)

    metrics = [accuracy, precision, recall, f1,]
    return metrics, predicciones_df, cm_df, class_df

def main(models_class1, path):
    print("🚀 Iniciando test...")

    datos = cargar_datos(path)

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

    X_test_processed = preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador)
    X_test_processed = X_test_processed[caracteritisticas_procesadas]
    print(f"✅ Preprocesamiento de test finalizado: {X_test_processed.shape[0]} filas, {X_test_processed.shape[1]} columnas.")

    output_dir = "datos/preprocesados"
    guardar_datos(X_test_processed,
                  output_dir)

    metrics_class1, predicciones_df, cm_df_class1, class_df_class1 = clasificacion_multiclase_tipo(y_test_class1, X_test_processed, models_class1)

    guardar_conf(path, metrics_class1, cm_df_class1, class_df_class1)
    
    print("🎯 Test finalizado.\n")