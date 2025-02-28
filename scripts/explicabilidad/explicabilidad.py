import io
import os
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt


def cargar_datos():
    """Carga todos los archivos procesados y los devuelve como DataFrames."""
    carpeta = r"datos/preprocesados"
    datos = {}

    # Cargar X_train
    path_X_train = os.path.join(carpeta, "X_train.csv")
    try:
        datos["X_train"] = pd.read_csv(path_X_train, low_memory=False)
        print(f"✅ X_train cargado: {datos['X_train'].shape[0]} filas, {datos['X_train'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_train}.")
        return None

    # Cargar X_val
    path_X_test = os.path.join(carpeta, "X_test_preprocesado.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None
    
    return datos
 
def guardar_exp(pre_path, exp, tipo):
    # Definir la ruta base
    dir_path = os.path.join("modelos", pre_path)
    os.makedirs(dir_path, exist_ok=True)  # 📌 Crear directorio si no existe

    if tipo == "lime":
        # Guardar la explicación en HTML
        path_html = os.path.join(dir_path, "lime_summary.html")
        exp.save_to_file(path_html)
        print(f"✅ Explicación LIME guardada en: {path_html}")

    elif tipo == "shap":
        # Guardar la imagen de SHAP en la misma carpeta
        path_shap = os.path.join(dir_path, "shap_summary.png")
        with open(path_shap, "wb") as f:
            f.write(exp.getbuffer())  # 📌 Guardar la imagen de SHAP
        print(f"✅ Gráfico SHAP guardado en: {path_shap}")

def metodo_lime(model_test, X_train, X_test, instancia):
    # Crear el explicador
    explainer = LimeTabularExplainer(X_train.values, 
                                    feature_names=X_train.columns.tolist(), 
                                    class_names=['Normal', 'Ataque'], 
                                    mode='classification',
                                    random_state=42)

    # Seleccionar una muestra de prueba
    exp = explainer.explain_instance(
        X_test.iloc[instancia].values, 
        lambda x: model_test.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    )
    
    return exp

def metodo_shape(model_test, X_test, instancia):
    
    if instancia is not None:
        # Crear el explicador específico para árboles de decisión
        explainer = shap.TreeExplainer(model_test)
        shap_values = explainer.shap_values(X_test)

        # Verificar estructura de shap_values
        # print(f"Tipo de shap_values: {type(shap_values)}")
        # print(f"Forma de shap_values: {shap_values.shape if hasattr(shap_values, 'shape') else 'No tiene shape'}")
        # print(f"Forma de X_test: {X_test.shape}")

        # Si shap_values tiene 3 dimensiones (ej: (123125, 34, 2)), seleccionar la clase 1 (Ataque)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # ✅ Tomar solo los valores de la clase 1 (Ataque)

        # Crear un objeto en memoria para almacenar la imagen
        img_buffer = io.BytesIO()

        # Crear figura sin mostrarla
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist(), show=False)
        
        # Guardar imagen en memoria en formato PNG
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
        plt.close()  # Cerrar la figura sin mostrarla

        # Posicionar el puntero al inicio del buffer para lectura
        img_buffer.seek(0)

        print("✅ Gráfico SHAP generado y guardado en memoria.")

        return img_buffer  # Devolver la imagen en un objeto BytesIO
    
    else:
        # Crear el explicador específico para árboles de decisión
        explainer = shap.TreeExplainer(model_test)
        shap_values = explainer.shap_values(X_test)

        # Verificar estructura de shap_values
        # print(f"Tipo de shap_values: {type(shap_values)}")
        # print(f"Forma de shap_values: {shap_values.shape if hasattr(shap_values, 'shape') else 'No tiene shape'}")
        # print(f"Forma de X_val: {X_val.shape}")

        # Seleccionar una sola instancia
        X_instance = X_test.iloc[[instancia]]
    
        # Si shap_values tiene 3 dimensiones (ej: (123125, 34, 2)), seleccionar la clase 1 (Ataque)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # ✅ Tomar solo los valores de la clase 1 (Ataque)

        # Seleccionar solo los SHAP values de la instancia específica
        shap_values_instance = shap_values[instancia]
    
        # Crear un objeto en memoria para almacenar la imagen
        img_buffer = io.BytesIO()

        # Crear figura sin mostrarla
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values_instance, 
                                         base_values=explainer.expected_value[1], 
                                         data=X_instance.values[0], 
                                         feature_names=X_instance.columns.tolist()))
        
        # Guardar imagen en memoria en formato PNG
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
        plt.close()  # Cerrar la figura sin mostrarla

        # Posicionar el puntero al inicio del buffer para lectura
        img_buffer.seek(0)

        print("✅ Gráfico SHAP generado y guardado en memoria.")

        return img_buffer  # Devolver la imagen en un objeto BytesIO

def main(model_test, path):  
    print("🚀 Iniciando explicabilidad...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_test = datos["X_test"]

    
    exp = metodo_lime(model_test, X_train, X_test, 9)
    # exp = metodo_shape(model_test, X_test, 1)
    
    guardar_exp(path, exp, "lime")
    
    print("🎯 Proceso de explicabilidad finalizado.")

    
