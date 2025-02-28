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

    class_report = classification_report(y_test_class3, y_pred_class3)

    return accuracy, precision, recall, f1, roc, class_report, y_pred_class3, probabilidades

def clasificacion_multiclase_categoria(y_pred_class3, y_test_class2, X_test_processed, model_class2):

        print("🔮 Clasificación multiclase (Categoría)...")
        # indices_test = np.where(y_pred_class3 == 1)[0]
        # X_test_processed = X_test_processed.iloc[indices_test]
        # y_test_class2 = y_test_class2.iloc[indices_test].values.ravel()

        y_test_class2 = y_test_class2.values.ravel()
        # Realizar predicciones
        y_pred_class2 = model_class2.predict(X_test_processed)

        probabilidades = model_class2.predict_proba(X_test_processed)

        # Evaluar el rendimiento
        accuracy = accuracy_score(y_test_class2, y_pred_class2)
        print(f'📈 Accuracy (Test): {accuracy:.4f}')

        precision = precision_score(y_test_class2, y_pred_class2, average='weighted', zero_division=0)
        print(f'📈 Precision (Test): {precision:.4f}')
        recall = recall_score(y_test_class2, y_pred_class2, average='weighted')
        print(f'📈 Recall (Test): {recall:.4f}')
        f1 = f1_score(y_test_class2, y_pred_class2, average='weighted')
        print(f'📈 F1 (Test): {f1:.4f}')

        class2_report = classification_report(y_test_class2, y_pred_class2, zero_division=0)
        return accuracy, class2_report, y_pred_class2, probabilidades

def clasificacion_multiclase_tipo(y_pred_class3, y_pred_class2, y_test_class1, X_test_processed, models_class1):

    print("🔮 Clasificación multiclase (Tipo)...")

    # indices_test = np.where(y_pred_class3 == 1)[0]
    # X_test_processed = X_test_processed.iloc[indices_test]
    # y_test_class1 = y_test_class1.iloc[indices_test].values.ravel()

    y_test_class1 = y_test_class1.values.ravel()


    model = models_class1["RDOS"]

    # Realizar predicciones
    y_pred_class1_cat = model.predict(X_test_processed)

    probabilidades = model.predict_proba(X_test_processed)

    # Obtener nombres de las clases correspondientes a las probabilidades
    class_labels = model.classes_

    # Asociar cada instancia con su clase y probabilidades
    probabilidades_con_nombres = [
        {class_labels[i]: prob[i] for i in range(len(class_labels))}
        for prob in probabilidades
    ]
    # Evaluar el rendimiento
    accuracy = accuracy_score(y_test_class1, y_pred_class1_cat)
    print(f'📈 Accuracy: {accuracy:.4f}')
    precision = precision_score(y_test_class1, y_pred_class1_cat, average='weighted', zero_division=0)
    print(f'📈 Precision: {precision:.4f}')
    recall = recall_score(y_test_class1, y_pred_class1_cat, average='weighted')
    print(f'📈 Recall: {recall:.4f}')
    f1 = f1_score(y_test_class1, y_pred_class1_cat, average='weighted')
    print(f'📈 F1: {f1:.4f}')



    return accuracy, y_pred_class1_cat

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


    accuracy, precision, recall, f1, roc, class_report, y_pred_class3, probabilidades_class3 = clasificacion_binaria(y_test_class3, X_test_processed, model)
    accuracy_class2, class2_report, y_pred_class2, probabilidades_class2 = clasificacion_multiclase_categoria(y_pred_class3, y_test_class2, X_test_processed, model_class2)
    accuracy_class1_tipo, y_pred_class1_cat = clasificacion_multiclase_tipo(y_pred_class3, y_pred_class2, y_test_class1, X_test_processed, models_class1)

    predicciones = [y_pred_class3, y_pred_class2, y_pred_class1_cat]
    # Suponiendo que predicciones es una lista de arrays
    predicciones_df = pd.DataFrame(predicciones).T  # Transponer para tener las predicciones como filas
    predicciones_df.columns = ['y_pred_class3', 'y_pred_class2', 'y_pred_class1_cat']  # Nombra las columnas si lo deseas

    # Guardar el DataFrame en un archivo CSV
    predicciones_df.to_csv('op2.csv', index=False)


    # guardar_conf(path, accuracy, precision, recall, f1, roc, class_report, accuracy_class2, accuracy_class1)
    guardar_conf(path, accuracy, precision, recall, f1, roc, class_report, accuracy_class2, class2_report)


    # instancia = 200

    # # Índice de la clase con mayor probabilidad en probabilidades_class2
    # indice_class2 = np.argmax(probabilidades_class2[instancia])
    # nombre_class2 = class_names[indice_class2]  # Obtener el nombre de la clase

    # # Imprimir resultados
    # print(nombre_class2)
    # # Obtener el diccionario de probabilidades para la instancia dada
    # probabilidades = probabilidades_class1_tipo[nombre_class2][instancia]
    # # Encontrar la clave con el valor máximo
    # categoria_maxima = max(probabilidades, key=probabilidades.get)
    # print(categoria_maxima)

    print("🎯 Test finalizado.\n")