import io
import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
from scripts.anomalias import anomalias
import datetime

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

    # Cargar X_test
    path_X_test = os.path.join(carpeta, "X_test_preprocesado.csv")
    try:
        datos["X_test"] = pd.read_csv(path_X_test, low_memory=False)
        print(f"✅ X_test cargado: {datos['X_test'].shape[0]} filas, {datos['X_test'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_test}.")
        return None
    
    
    # Cargar y_test_class1
    path_y_test_class1 = os.path.join(carpeta, "y_test_class1.csv")
    try:
        datos["y_test_class1"] = pd.read_csv(path_y_test_class1, low_memory=False)
        print(f"✅ y_test_class1 cargado: {datos['y_test_class1'].shape[0]} filas, {datos['y_test_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_test_class1}.")
        return None
    
    return datos
 
def guardar_exp(pre_path, exp, tipo, model_test=None, X_test=None, instancia=None):
    
    # Definir la ruta base
    dir_path = os.path.join("modelos", pre_path)
    os.makedirs(dir_path, exist_ok=True)

    if tipo == "lime":
        path_html = os.path.join(dir_path, "lime_summary.html")
        lime_html = exp.as_html()

        fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        # Determinar si es ataque o no
        es_ataque = None
        if model_test and X_test is not None and instancia is not None:
            clase_predicha = model_test.predict(X_test.iloc[[instancia]])[0]
            es_ataque = clase_predicha != "Normal"


        html_final = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Explicación LIME</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                :root {{
                    --verde-claro: #D5E8D4;
                    --rosado-claro: #F8CECC;
                    --azul-claro: #DAE8FC;
                    --gris-claro: #CCCCCC;
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: var(--verde-claro);
                    color: #333;
                }}
                header {{
                    background-color: var(--gris-claro);
                    padding: 20px 40px;
                    text-align: center;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                }}
                header h1 {{
                    margin: 0;
                    font-size: 28px;
                    color: #2F4F4F;
                }}
                header p {{
                    margin: 5px 0 0;
                    font-size: 14px;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 40px auto;
                    background-color: #fff;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    font-size: 20px;
                    margin-bottom: 10px;
                    color: #2F4F4F;
                }}
                .section p {{
                    font-size: 15px;
                    line-height: 1.6;
                }}
                .alert {{
                    padding: 15px 20px;
                    font-size: 15px;
                    font-weight: bold;
                    border-radius: 6px;
                    margin-top: 10px;
                }}
                .alert.ataque {{
                    background-color: var(--rosado-claro);
                    border-left: 6px solid #c62828;
                    color: #c62828;
                }}
                .alert.normal {{
                    background-color: var(--verde-claro);
                    border-left: 6px solid #388E3C;
                    color: #2E7D32;
                }}
                .lime-output {{
                    margin-top: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    font-size: 14px;
                }}
                th, td {{
                    text-align: left;
                    padding: 10px;
                    border: 1px solid #ccc;
                }}
                th {{
                    background-color: var(--gris-claro);
                }}
                tr:nth-child(even) {{
                    background-color: var(--azul-claro);
                }}
                tr:nth-child(odd) {{
                    background-color: var(--gris-claro);
                }}
                .footer {{
                    text-align: center;
                    margin: 40px 0 10px;
                    font-size: 12px;
                    color: #888;
                }}
                @media (max-width: 768px) {{
                    .container {{
                        margin: 20px;
                        padding: 20px;
                    }}
                    header h1 {{
                        font-size: 22px;
                    }}
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>📊 Explicación de Modelo con LIME</h1>
                <p>Generado el {fecha}</p>
            </header>

            <div class="container">
                <div class="section">
                    <h2>Resultado de la predicción</h2>
                    {"<div class='alert ataque'>⚠️ Ataque detectado</div>" if es_ataque else "<div class='alert normal'>✅ Tráfico normal</div>"}
                </div>

                <div class="section">
                    <h2>Explicación generada por LIME</h2>
                    <p>A continuación se muestra una explicación local del modelo sobre la instancia seleccionada, indicando qué características han influido más en la predicción.</p>
                    <div class="lime-output">
                        {lime_html}
                    </div>
                </div>
            </div>

            <div class="footer">
                Explicabilidad generada con LIME | Proyecto Rafael Sánchez Navarro
            </div>
        </body>
        </html>
        """



        with open(path_html, "w", encoding="utf-8") as f:
            f.write(html_final)

        print(f"✅ HTML LIME estilizado guardado en: {path_html}")
        
        

        # Guardar gráfico estático de LIME como imagen con separación vertical y ancho reducido
        label_explicado = exp.available_labels()[0]
        labels, weights = zip(*exp.as_list(label=label_explicado))
        print(labels)
        print(weights)
        
        # Figura más estrecha (menos ancho) y más alta
        plt.figure(figsize=(8, 8))  # ancho reducido de 10 a 6, altura aumentada

        colors = ['#F8CECC' if w < 0 else '#D5E8D4' for w in weights]

        bar_positions = range(len(labels))
        bar_height = 0.8  # más separación vertical

        plt.barh(
            bar_positions,
            weights,
            color=colors,
            edgecolor='grey',
            height=bar_height
        )

        plt.yticks(bar_positions, labels, fontsize=10)
        plt.xlabel("Importancia", fontsize=10)
        plt.title("Explicación LIME", fontsize=10)
        plt.xticks(fontsize=10)
        plt.xlim(-0.0035, 0.0035)  # límite del eje X fijo
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.2)  # más espacio para etiquetas
        plt.tight_layout(pad=2)

        # Guardar imagen
        path_img = os.path.join(dir_path, "lime_summary.png")
        plt.savefig(path_img, dpi=300)
        plt.close()

        print(f"✅ Imagen LIME más estrecha guardada en: {path_img}")



    elif tipo == "shap":
        # Guardar la imagen de SHAP en la misma carpeta
        path_shap = os.path.join(dir_path, "shap_summary.png")
        with open(path_shap, "wb") as f:
            f.write(exp.getbuffer())  # 📌 Guardar la imagen de SHAP
        print(f"✅ Gráfico SHAP guardado en: {path_shap}")

def metodo_lime(model_test, X_train, X_test, instancia):
    """
    Explicación LIME para clasificación multiclase real (incluye tipos específicos de ataque).
    """
    # Obtener nombres de clases directamente desde el modelo
    class_names = list(model_test.classes_)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode='classification',
        random_state=42
    )

    # Usar directamente predict_proba del modelo sin convertir a binaria
    exp = explainer.explain_instance(
        data_row=X_test.iloc[instancia].values,
        predict_fn=lambda x: model_test.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
        top_labels=1,
        num_features=17,
    )

    return exp

def metodo_shap(model_test, X_test, instancia=None):
    """
    Genera una visualización SHAP para una instancia específica (waterfall) o resumen general (summary_plot).
    """
    explainer = shap.TreeExplainer(model_test)
    shap_values = explainer.shap_values(X_test)

    # Si es multiclase, shap_values será una lista (una por clase)
    multiclase = isinstance(shap_values, list)

    # Crear buffer de imagen en memoria
    img_buffer = io.BytesIO()

    if instancia is not None:
        # ✅ Explicación individual - waterfall plot
        X_instance = X_test.iloc[[instancia]]
        clase_predicha = model_test.predict(X_instance)[0]

        # Obtener índice de la clase predicha
        clases = model_test.classes_
        class_index = list(clases).index(clase_predicha)

        # Seleccionar los valores SHAP de esa clase
        valores_instancia = shap_values[class_index][instancia] if multiclase else shap_values[instancia]

        # Crear y guardar gráfico waterfall
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(
            values=valores_instancia,
            base_values=explainer.expected_value[class_index] if multiclase else explainer.expected_value,
            data=X_instance.values[0],
            feature_names=X_instance.columns.tolist()
        ))
    else:
        # ✅ Resumen general - summary plot
        clase_default = 1 if multiclase else None
        valores = shap_values[clase_default] if multiclase else shap_values

        # Crear y guardar gráfico summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(valores, X_test, feature_names=X_test.columns.tolist(), show=False)

    # Guardar la imagen
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    img_buffer.seek(0)

    print("✅ Gráfico SHAP generado y guardado en memoria.")
    return img_buffer

def main(model_test, path):  
    print("🚀 Iniciando explicabilidad...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_test = datos["X_test"]
    y_test_class1 = datos["y_test_class1"]
    
    list = ['read_write_physical.process_scaled', 'Duration_scaled', 'total_bytes_scaled', 'Avg_num_cswch/s_scaled', 'Service_dns', 'Avg_system_time_scaled', 'paket_rate_scaled', 'is_with_payload_scaled', 'Avg_user_time_scaled', 'Avg_ideal_time_scaled', 'total_packet_scaled', 'Service_other', 'Service_mqtt', 'Service_coap', 'std_num_cswch/s_scaled', 'is_syn_only_scaled', 'Protocol_udp', 'Conn_state_scaled', 'Avg_nice_time_scaled', 'Login_attempt_scaled']

    X_test = X_test[list]
    
    instancia = 62793
    
    print(y_test_class1.iloc[instancia])
    exp = metodo_lime(model_test, X_train, X_test, instancia)
    # exp = metodo_shape(model_test, X_test, 1)
    
    guardar_exp(path, exp, "lime", model_test=model_test, X_test=X_test, instancia=instancia)

    print("🎯 Proceso de explicabilidad finalizado.")

    
