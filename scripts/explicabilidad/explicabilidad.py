import io
import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
from scripts.anomalias import anomalias
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.image as mpimg



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
 
def guardar_exp(pre_path, exp, model_test=None, X_test=None, instancia=None):
    
    # Definir la ruta base
    dir_path = os.path.join("modelos", pre_path)
    os.makedirs(dir_path, exist_ok=True)

    
    path_html = os.path.join(dir_path, "lime_summary.html")
    lime_html = exp.as_html()

    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    probs_html = ""
    if model_test and X_test is not None and instancia is not None:
        probs = model_test.predict_proba(X_test.iloc[[instancia]])[0]
        clases = model_test.classes_
        probs_dict = {clase: f"{prob:.2%}" for clase, prob in zip(clases, probs)}

        rows = "".join(f"<tr><td>{clase}</td><td>{prob}</td></tr>" for clase, prob in probs_dict.items())
        probs_html = f"""
        <div class="section">
            <h2>Probabilidades por clase</h2>
            <table>
                <tr><th>Clase</th><th>Probabilidad</th></tr>
                {rows}
            </table>
        </div>
        """

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

            {probs_html}
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
    
    

    # Obtener clase predicha y su probabilidad
    clase_predicha = model_test.predict(X_test.iloc[[instancia]])[0]
    probs = model_test.predict_proba(X_test.iloc[[instancia]])[0]
    clases = model_test.classes_
    pred_idx = list(clases).index(clase_predicha)
    prob_predicha = probs[pred_idx]

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
    plt.xlabel("Contribución de cada característica", fontsize=10)
    plt.title(f"Explicación LIME - Clase: {clase_predicha} ({prob_predicha:.2%})", fontsize=10)
    plt.xticks(fontsize=10)
    max_weight = max(abs(w) for w in weights)
    plt.xlim(-max_weight * 1.2, max_weight * 1.2)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.2)  # más espacio para etiquetas
    plt.tight_layout(pad=2)

    # Guardar imagen
    path_img = os.path.join(dir_path, "lime_summary.png")
    plt.savefig(path_img, dpi=300)
    plt.close()

    print(f"✅ Imagen LIME más estrecha guardada en: {path_img}")

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
        num_features=34,
    )

    return exp

def metodo_shap_barplot(model, X, instancia, pre_path):

    if hasattr(model, "feature_names_in_"):
        X_input = X[model.feature_names_in_]
    else:
        X_input = X

    X_inst = X_input.iloc[[instancia]]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_inst)

    pred = model.predict(X_inst)[0]
    prob = model.predict_proba(X_inst)[0]
    class_index = list(model.classes_).index(pred)
    pred_prob = prob[class_index]

    # --- Seleccionar valores SHAP correctamente según formato multiclase
    if isinstance(shap_values, list):
        shap_instance_values = shap_values[class_index][0]
    else:
        shap_instance_values = shap_values[0][np.arange(X_input.shape[1]), class_index]

    shap_instance_values = np.array(shap_instance_values).flatten()

    if len(shap_instance_values) != X_input.shape[1]:
        raise ValueError(f"❌ Mismatch: {len(shap_instance_values)} SHAP values vs {X_input.shape[1]} features")

    # Discretizar los valores de entrenamiento para generar etiquetas tipo LIME
    discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    X_discretized = discretizer.fit_transform(X_input)

    # Obtener límites de los bins
    bin_edges = discretizer.bin_edges_

    # Construir etiquetas estilo LIME con rangos
    labels = []
    for i, col in enumerate(X_input.columns):
        val = X_inst.iloc[0, i]
        edges = bin_edges[i]

        # Buscar el bin al que pertenece el valor
        bin_idx = np.digitize(val, edges, right=True) - 1
        bin_idx = max(0, min(bin_idx, len(edges) - 2))  # evitar overflow

        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]

        # Formato estilo LIME
        if val <= lower:
            label = f"{col} <= {lower:.2f}"
        elif val > upper:
            label = f"{col} > {upper:.2f}"
        else:
            label = f"{lower:.2f} < {col} <= {upper:.2f}"

        labels.append(label)

    values = shap_instance_values
    print(labels)
    print(values)
    
    # Ordenar por magnitud absoluta descendente
    sorted_indices = np.argsort(-np.abs(values))
    values = values[sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    colors = ['#F8CECC' if v < 0 else '#D5E8D4' for v in values]
    bar_positions = range(len(labels))

    plt.figure(figsize=(8, 8))
    plt.barh(bar_positions, values, color=colors, edgecolor='grey', height=0.8)
    
    plt.yticks(bar_positions, labels, fontsize=10)
    plt.xlabel("Contribución de cada característica", fontsize=10)
    plt.title(f"Explicación SHAP - Clase: {pred} ({pred_prob:.2%})", fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlim(-max(abs(values)) * 1.2, max(abs(values)) * 1.2)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout(pad=2)
    
    # Construir la ruta de salida usando pre_path directamente
    output_path = os.path.join("modelos", pre_path, "shap_barplot.png")

    # Guardar la figura
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ SHAP barplot guardado en: {output_path}")

def analizar_modelo_con_shap(model, X, n_instancias=50, path=""):
    output_dir = os.path.join("modelos", path)
    os.makedirs(output_dir, exist_ok=True)

    X_sample = X.iloc[:n_instancias].copy()
    n_features = X_sample.shape[1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.gcf().set_size_inches(20, 14)

    # === summary_plot
    if isinstance(shap_values, list):  # multiclase
        shap.summary_plot(shap_values[0], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)

    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"✅ summary_plot guardado en: {summary_path}")
      
def main(model_test, path):  
    print("🚀 Iniciando explicabilidad...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_test = datos["X_test"]
    y_test_class1 = datos["y_test_class1"]

    instancia = 122178
    
    print(y_test_class1.iloc[instancia])
    exp = metodo_lime(model_test, X_train, X_test, instancia)
    guardar_exp(path, exp, model_test=model_test, X_test=X_test, instancia=instancia)
    metodo_shap_barplot(model_test, X_test, instancia, path)
    # analizar_modelo_con_shap(model_test, X_test, 10, path)
    

    ruta_base = os.path.join("modelos", path)
    lime_path = os.path.join(ruta_base, "lime_summary.png")
    shap_path = os.path.join(ruta_base, "shap_barplot.png")

    # Cargar imágenes desde su ruta
    img1 = mpimg.imread(shap_path)
    img2 = mpimg.imread(lime_path)

    # Crear nueva figura combinada
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img1)
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()

    # Guardar imagen combinada en la misma carpeta
    output_path = os.path.join(ruta_base, "explicabilidad_combinada.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Imagen combinada guardada en: {output_path}")

    print("🎯 Proceso de explicabilidad finalizado.")

    
