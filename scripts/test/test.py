import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
from scripts.preprocesamiento.conversion import delete_ip_port, fix_dtype

def cargar_datos(pre_path):
    """Carga todos los archivos procesados, el preprocesamiento y los devuelve como un diccionario."""
    carpeta = r"datos/preprocesados"
    datos = {}

    # Cargar X_test
    path_X_test = os.path.join(carpeta, "X_test.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None

    # Cargar y_test_class3
    path_y_test_class3 = os.path.join(carpeta, "y_test_class3.csv")
    try:
        datos["y_test_class3"] = pd.read_csv(path_y_test_class3, low_memory=False)
        print(f"✅ y_test_class3 cargado: {datos['y_test_class3'].shape[0]} filas, {datos['y_test_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class3}.")
        return None
    
    # Cargar y_test_class2
    path_y_test_class2 = os.path.join(carpeta, "y_test_class2.csv")
    try:
        datos["y_test_class2"] = pd.read_csv(path_y_test_class2, low_memory=False)
        print(f"✅ y_test_class2 cargado: {datos['y_test_class2'].shape[0]} filas, {datos['y_test_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class2}.")
        return None
    
    # Cargar y_test_class2
    path_y_test_class1 = os.path.join(carpeta, "y_test_class1.csv")
    try:
        datos["y_test_class1"] = pd.read_csv(path_y_test_class1, low_memory=False)
        print(f"✅ y_test_class1 cargado: {datos['y_test_class1'].shape[0]} filas, {datos['y_test_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class1}.")
        return None

    # Cargar componentes de preprocesamiento
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

def guardar_conf(pre_path, accuracy, precision, recall, f1, roc, class_report, accuracy_class2, class2_report):
    # Guardar resumen
    resumen = crear_resumen(accuracy, precision, recall, f1, roc, class_report, accuracy_class2, class2_report)
    path_resumen = os.path.join(f"modelos/{pre_path}", "resumen_test.txt")
    with open(path_resumen, "w", encoding="utf-8") as f:
        f.write(resumen)

def guardar_datos(X_test_processed, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    X_test_processed.to_csv(os.path.join(output_dir, "X_test_preprocesado.csv"), index=False)

def crear_resumen(accuracy, precision, recall, f1_score, roc, class_report, accuracy_class2, class2_report):
    texto = f"Accuracy: {accuracy:.4f}\n"
    texto += f"Precision: {precision:.4f}\n"
    texto += f"Recall: {recall:.4f}\n"
    texto += f"F1 Score: {f1_score:.4f}\n"
    texto += f"ROC AUC: {roc:.4f}\n\n"
    texto += class_report
    texto += f"\n Accuracy Categoria: {accuracy_class2:.4f}\n\n"
    texto += class2_report

    return texto 

def preprocesamiento_test(X_test, imputador_cat, imputador_num, normalizacion, discretizador, decodificador):
    
    # Identificar columnas categóricas, numéricas y booleanas
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

    # Convertir las matrices escaladas a DataFrames
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=[f"{col}_scaled" for col in numerical_cols], index=X_test.index)

    ##############################################################################
    
    if discretizador is not None:
        X_test_discrete = discretizador.transform(X_test[numerical_cols])

        # Convertir las matrices discretizadas a DataFrames
        X_test_discretized_df = pd.DataFrame(X_test_discrete, columns=[f"{col}_discrete" for col in numerical_cols], index=X_test.index)
    
    ##############################################################################
    
    if discretizador is not None:
        processed_numeric_test = pd.concat([X_test_scaled_df, X_test_discretized_df], axis=1)
    else:
        processed_numeric_test = X_test_scaled_df
    
    ##############################################################################
    
    X_test_encoded = decodificador.transform(X_test[categorical_cols])

    # Obtener los nombres de las nuevas columnas codificadas
    encoded_cols = decodificador.get_feature_names_out(categorical_cols)

    # Convertir las matrices codificadas a DataFrames
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
    
    ##############################################################################
    
    # Combinar con las características categóricas codificadas
    X_test_processed = pd.concat([processed_numeric_test, X_test_encoded_df], axis=1)
    
    # Opcional: Reordenar las columnas si es necesario
    X_test_processed = X_test_processed.reindex(sorted(X_test_processed.columns), axis=1)
    
    return X_test_processed 
   
def graficar_roc(y_test_class3, y_pred_class3):
    
    # Calcular las probabilidades de la clase positiva (por ejemplo, clase 1)
    fpr, tpr, thresholds = roc_curve(y_test_class3, y_pred_class3)

    # Calcular el área bajo la curva ROC
    roc_auc = roc_auc_score(y_test_class3, y_pred_class3)

    # Dibujar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
 
def clasificacion_binaria(y_test_class3, X_test_processed, model):
    print("🔮 Clasificación binaria...")
    # Realizar predicciones
    y_pred_class3 = model.predict(X_test_processed)

    # probabilidades = suavizar_probabilidades(model.predict_proba(X_test_processed))
    probabilidades = model.predict_proba(X_test_processed)
    
    accuracy = accuracy_score(y_test_class3, y_pred_class3)
    print(f"📈 Accuracy (Test): {accuracy:.4f}")
    
    precision = precision_score(y_test_class3, y_pred_class3)
    print(f"📈 Precision (Test): {precision:.4f}")
    
    recall = recall_score(y_test_class3, y_pred_class3)
    print(f"📈 Recall (Test): {recall:.4f}")
    
    f1 = f1_score(y_test_class3, y_pred_class3)
    print(f"📈 F1 (Test): {f1:.4f}")
    
    roc = roc_auc_score(y_test_class3, y_pred_class3)
    print(f'📈 ROC (Test): {roc:.4f}')
    
    # graficar_roc(y_test_class3, y_pred_class3)
    
    predicciones_df = pd.DataFrame({'Predicciones': y_pred_class3})
    predicciones_df.to_csv('predicciones_multiclase.csv', index=False)

    class_report = classification_report(y_test_class3, y_pred_class3)
 
    return accuracy, precision, recall, f1, roc, class_report, y_pred_class3, probabilidades
 
def clasificacion_multiclase_categoria(y_pred_class3, y_test_class2, X_test_processed, model_class2):

        print("🔮 Clasificación multiclase (Categoría)...")
        indices_test = np.where(y_pred_class3 == 1)[0]
        X_test_processed = X_test_processed.iloc[indices_test]
        y_test_class2 = y_test_class2.iloc[indices_test].values.ravel()
        
        # Realizar predicciones
        y_pred_class2 = model_class2.predict(X_test_processed)
        
        # probabilidades = model_class2.predict_proba(X_test_processed)
        probabilidades = suavizar_probabilidades(model_class2.predict_proba(X_test_processed)) 

        # Inicializar el array de probabilidades completas con ceros
        probabilidades_completas = np.zeros((len(y_pred_class3), probabilidades.shape[1]))

        # Asignar probabilidades solo en los índices correspondientes
        probabilidades_completas[indices_test] = probabilidades
        
        # Obtener los nombres de las clases
        clases_nombres = model_class2.classes_
        probabilidades_dict = [
            {clase: prob for clase, prob in zip(clases_nombres, probs)}
            for probs in probabilidades_completas
        ]
        
        # Evaluar el rendimiento
        accuracy = accuracy_score(y_test_class2, y_pred_class2)
        print(f'📈 Accuracy (Test): {accuracy:.4f}')
        
        precision = precision_score(y_test_class2, y_pred_class2, average='weighted', zero_division=0)
        print(f'📈 Precision (Test): {precision:.4f}')
        recall = recall_score(y_test_class2, y_pred_class2, average='weighted')
        print(f'📈 Recall (Test): {recall:.4f}')
        f1 = f1_score(y_test_class2, y_pred_class2, average='weighted')
        print(f'📈 F1 (Test): {f1:.4f}')
        
        # Cargar el CSV existente
        predicciones_df = pd.read_csv('predicciones_multiclase.csv')
        
        # Añadir las nuevas predicciones como una nueva columna
        predicciones_df['Predicciones_Class2'] = "Normal"  # Inicializa la columna con NaN
        predicciones_df.loc[indices_test, 'Predicciones_Class2'] = y_pred_class2  # Asigna las predicciones
        
        # Guardar el DataFrame actualizado en el CSV
        predicciones_df.to_csv('predicciones_multiclase.csv', index=False)
    
        class2_report = classification_report(y_test_class2, y_pred_class2, zero_division=0)
        
        # Completar con "NORMAL" para los índices que no están en 'indices_test'
        y_pred_class2_full = np.full_like(y_pred_class3, "Normal", dtype=object)
        # Asignar las predicciones de la clase 2 a los índices correspondientes
        y_pred_class2_full[indices_test] = y_pred_class2
        
        return accuracy, class2_report, y_pred_class2_full, probabilidades_dict
 
def clasificacion_multiclase_tipo(y_pred_class2, y_test_class1, X_test_processed, models_class1):
    print("🔮 Clasificación multiclase (Tipo)...")
    
    indices_test = np.where(y_pred_class2 != "Normal")[0]
    indice_test_normal = np.where(y_pred_class2 == "Normal")[0]
    indices_total = np.concatenate((indices_test, indice_test_normal))
    indices_total = np.sort(indices_total)
    y_test_class1 = y_test_class1.values.ravel()
    
    class_names = ['Exploitation', 'Lateral _movement', 'Reconnaissance', 'Tampering', 'Weaponization']
    # class_names = np.unique(y_pred_class2)
    
    probabilidades_tipo = [{} for _ in range(len(y_pred_class2))]

    accuracy_total = 0
    y_pred_class1_total = []  # Lista para almacenar las predicciones de clase 1
    
    y_pred_class1_total = ["Normal"] * len(y_pred_class2)  # Inicializa con "Normal"
    # Leer el CSV una vez
    predicciones_df = pd.read_csv('predicciones_multiclase.csv')
    
    
    for name in class_names:

        print(f"🔮 Clasificación multiclase (Tipo) para {name}...")
        indices_test_cat = np.where(y_pred_class2 == name)[0]
        X_test_processed_cat = X_test_processed.iloc[indices_test_cat]
        y_test_class1_cat = y_test_class1[indices_test_cat]
        model = models_class1[name]
        class_labels = model.classes_  
        print(class_labels)
        
        # Realizar predicciones
        y_pred_class1_cat = model.predict(X_test_processed_cat)
        
        # Asignar las predicciones a la lista total usando el índice correcto
        for idx, pred in zip(indices_test_cat, y_pred_class1_cat):
            y_pred_class1_total[idx] = pred
        
        # probabilidades = model.predict_proba(X_test_processed_cat)
        probabilidades = suavizar_probabilidades(model.predict_proba(X_test_processed_cat)) 
        
        for i, idx in enumerate(indices_test_cat):
            # Crear un diccionario con las clases y sus probabilidades
            probs_dict = {class_labels[j]: float(probabilidades[i][j]) for j in range(len(class_labels))}
            probabilidades_tipo[idx] = probs_dict


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_test_class1_cat, y_pred_class1_cat)
        print(f'📈 Accuracy: {accuracy:.4f}')
        precision = precision_score(y_test_class1_cat, y_pred_class1_cat, average='weighted', zero_division=0)
        print(f'📈 Precision: {precision:.4f}')
        recall = recall_score(y_test_class1_cat, y_pred_class1_cat, average='weighted')
        print(f'📈 Recall: {recall:.4f}')
        f1 = f1_score(y_test_class1_cat, y_pred_class1_cat, average='weighted')
        print(f'📈 F1: {f1:.4f}')
        
        accuracy_total = accuracy_total + accuracy
        
            
        # Actualizar el DataFrame con las predicciones en cada iteración
        predicciones_df.loc[indices_total, 'Predicciones_Class1'] = y_pred_class1_total 
        
        
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
    predicciones_df.to_csv('predicciones_multiclase.csv', index=False)
            
    return accuracy, probabilidades_tipo, class_names

def suavizar_probabilidades(probabilidades, epsilon=0.01):
    return (probabilidades + epsilon) / (1 + epsilon * probabilidades.shape[1])

def main(model, path, model_class2, models_class1):  
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

    accuracy, precision, recall, f1, roc, class_report, y_pred_class3, probabilidades_class3 = clasificacion_binaria(y_test_class3, X_test_processed, model)
    accuracy_class2, class2_report, y_pred_class2, probabilidades_class2 = clasificacion_multiclase_categoria(y_pred_class3, y_test_class2, X_test_processed, model_class2)
    accuracy_class1_tipo, probabilidades_class1_tipo, class_names = clasificacion_multiclase_tipo(y_pred_class2, y_test_class1, X_test_processed, models_class1)
    
    
    
    # guardar_conf(path, accuracy, precision, recall, f1, roc, class_report, accuracy_class2, accuracy_class1)
    guardar_conf(path, accuracy, precision, recall, f1, roc, class_report, accuracy_class2, class2_report)

    
    instancia = 1
    
    
    if np.max(probabilidades_class3[instancia][1]) > 0.5:
        print("🔴 Anomalía detectada.")
        max_prob_class3 = np.max(probabilidades_class3[instancia])
        max_index_class3 = np.argmax(probabilidades_class3[instancia])
        
        max_clase_class2 = max(probabilidades_class2[instancia], key=probabilidades_class2[instancia].get)
        print(probabilidades_class2[instancia])
        max_prob_class2 = probabilidades_class2[instancia][max_clase_class2]
        
        if max_clase_class2 not in ["RDOS", "Exfiltration", "C&C", "crypto-ransomware"]:
            max_clase_class1 = max(probabilidades_class1_tipo[instancia], key=probabilidades_class1_tipo[instancia].get)
            print(probabilidades_class1_tipo[instancia])
            max_prob_class1 = probabilidades_class1_tipo[instancia][max_clase_class1]
        else:
            max_clase_class1 = max_clase_class2
            max_prob_class1 = 1
    else:
        max_index_class3 = 0
        max_prob_class2 = 0
        max_prob_class1 = 0
        
        max_prob_class3 = 0
        max_clase_class2 = "Normal"
        max_clase_class1 = "Normal"
        
        print("🟢 No se detectó anomalía.")
        
    
    resultado = max_index_class3 * max_prob_class2 * max_prob_class1
    print(max_index_class3, max_clase_class2, max_clase_class1)
    print(max_prob_class3, max_prob_class2, max_prob_class1)
    print(f"Total: {resultado}")

    print("🎯 Test finalizado.\n")