from datetime import datetime
import os
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from scripts.preprocesamiento import preprocesamiento
from scripts.preprocesamiento.preprocesamiento_utils import discretizers, scalers, imputers, encoders
from scripts.preprocesamiento.reduccion_dimensionalidad import seleccionar_variables_pca, seleccionar_variables_randomForest, seleccionar_variables_rfe, proyectar_tsne

from scripts.entrenamiento import entrenamientoNoSupervisado, entrenamientoSupervisado
from scripts.entrenamiento.entrenamiento_utils.validacion import validation_methods
from modelos.diccionario_modelos import algorithms

from scripts.test import test


def guardar_conf(model, accuracy, precision, recall, f1, roc, imputador_cat, imputador_num, 
                 normalizacion, discretizador, decodificador, 
                 caracteritisticas_seleccionadas, caracteritisticas_procesadas, grid, random_grid, validacion_grid, ensemble, reduccion_dimensionalidad):
    
    # Crear la carpeta si no existe
    output_dir = f"modelos/{model.__class__.__name__}_{accuracy:.4f}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar los imputadores y demás en la misma carpeta si existen
    if imputador_cat is not None:
        path_imputador_cat = os.path.join(output_dir, "imputador_cat.pkl")
        joblib.dump(imputador_cat, path_imputador_cat)

    if imputador_num is not None:
        path_imputador_num = os.path.join(output_dir, "imputador_num.pkl")
        joblib.dump(imputador_num, path_imputador_num)

    if normalizacion is not None:
        path_normalizacion = os.path.join(output_dir, "normalizacion.pkl")
        joblib.dump(normalizacion, path_normalizacion)

    if discretizador is not None:
        path_discretizador = os.path.join(output_dir, "discretizador.pkl")
        joblib.dump(discretizador, path_discretizador)

    if decodificador is not None:
        path_decodificador = os.path.join(output_dir, "decodificador.pkl")
        joblib.dump(decodificador, path_decodificador)

    if caracteritisticas_seleccionadas is not None:
        path_reduccion = os.path.join(output_dir, "caracteritisticas_seleccionadas.pkl")
        joblib.dump(caracteritisticas_seleccionadas, path_reduccion)
        
    if caracteritisticas_procesadas is not None:
        path_reduccion = os.path.join(output_dir, "caracteritisticas_procesadas.pkl")
        joblib.dump(caracteritisticas_procesadas, path_reduccion)
        
        
    # Guardar el modelo
    nombre_modelo = f"{model.__class__.__name__}_{accuracy:.4f}.pkl"
    path = os.path.join(output_dir, nombre_modelo)
    joblib.dump(model, path)
    
    # Guardar resumen
    resumen = crear_resumen(model, accuracy, precision, recall, f1, roc, imputador_cat, imputador_num, 
                            normalizacion, discretizador, decodificador, caracteritisticas_procesadas, grid, random_grid, 
                            validacion_grid, ensemble, reduccion_dimensionalidad)
    
    path_resumen = os.path.join(output_dir, "resumen_train.txt")
    with open(path_resumen, "w", encoding="utf-8") as f:
        f.write(resumen)
        
    print(f"📁 Pipeline guardado en: {output_dir}\n")
    
def crear_resumen(model_train, accuracy, precision, recall, f1_score, roc, imputador_cat, imputador_num, normalizacion, 
                  discretizador, decodificador, caracteritisticas_procesadas, grid, random_grid, validacion_grid, ensemble, reduccion_dimensionalidad):
    
    cantidad = len(caracteritisticas_procesadas)
    texto = f"Fecha: {datetime.now()}\n\n"
    
    if ensemble:
        texto += f"Modelo: {model_train.estimators}\n"
    else:
        texto += f"Modelo: {model_train.__class__.__name__}\n"
    
    texto += f"Accuracy: {accuracy:.4f}\n"
    texto += f"Precision: {precision:.4f}\n"
    texto += f"Recall: {recall:.4f}\n"
    texto += f"F1 Score: {f1_score:.4f}\n"
    texto += f"ROC AUC: {roc:.4f}\n\n"
    
    if imputador_cat is not None:
        texto += f"Imputador Categórico: {imputador_cat.__class__.__name__} (Estrategia: {imputador_cat.strategy})\n"
    else:
        texto += "Imputador Categórico: Ninguno\n"

    if imputador_num is not None:
        texto += f"Imputador Numérico: {imputador_num.__class__.__name__} (Estrategia: {imputador_num.strategy})\n"
    else:
        texto += "Imputador Numérico: Ninguno\n"
    texto += "Normalización: " + (normalizacion.__class__.__name__ if normalizacion is not None else "Ninguno") + "\n"
    texto += "Discretizador: " + (discretizador.__class__.__name__ if discretizador is not None else "Ninguno") + "\n"
    texto += "Decodificador: " + (decodificador.__class__.__name__ if decodificador is not None else "Ninguno") + "\n"
    if reduccion_dimensionalidad is not None:
        texto += "Reducción de dimensionalidad: " + (reduccion_dimensionalidad.__name__ if reduccion_dimensionalidad is not None else "Ninguno") + "\n"
    texto += f"Características: {cantidad} \n"
    texto += "\nGrid Search: " + ("Sí" if grid else "No") + "\n"
    if grid:
        texto += "Random Grid Search: " + ("Sí" if random_grid else "No") + "\n"
        texto += "Validación: " + (validacion_grid.__class__.__name__ if validacion_grid is not None else "Ninguno") + "\n"
    
    return texto 

def main():
    random_state = 42
    
    # Preprocesamiento ####################################################################
    
    imputador_cat= imputers.imputers['categorical']['most_frequent']
    imputador_num = imputers.imputers['numeric']['mean']
    normalizacion = scalers.scalers['robust']
    discretizador = None
    decodificador = encoders.encoders['one_hot']
    reduccion_dimensionalidad = seleccionar_variables_pca
    
    caracteritisticas_seleccionadas, caracteritisticas_procesadas = preprocesamiento.main(
                        random_state,
                        imputador_cat, 
                        imputador_num, 
                        normalizacion, 
                        discretizador, 
                        decodificador, 
                        reduccion_dimensionalidad
    )
    
    
    # Eleccion del modelo ###################################################################
    
    ensemble = False
    
    # clf1 = algorithms['DecisionTreeClassifier'](random_state=random_state)
    # clf2 = algorithms['GaussianNB']()
    # model = VotingClassifier(
    #     estimators=[
    #             ('mwbp', clf1), # Modelo baso en arboles
    #             ('gnb', clf2),
    #             # añadir mas diversidad de algoritmos que no esten basados en arboles
    #         ],
    #         voting='soft'  # Cambiar a 'hard' para votación mayoritaria
    #     )
    
    # print(f"➡️  Ensemble configurado con los clasificadores: {[nombre for (nombre, _) in model.estimators]}\n")
    
    model = algorithms['SVC'](random_state=random_state) # Poner semilla a los que la necesiten
    
    
    # Entrenamiento #########################################################################
    
    grid = False
    grid_n_iter = 10
    random_grid = True
    validacion_grid = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    
    model_train, accuracy, precision, recall, f1, roc  = entrenamientoSupervisado.main(
                        random_state,
                        model,
                        grid,
                        validacion_grid,
                        grid_n_iter,
                        random_grid,
                        ensemble
    )
        
    # Guardar modelo #######################################################################
    
    # guardar_conf(model_train, accuracy)
    guardar_conf(model_train, accuracy, precision, recall, f1, roc, imputador_cat, 
                 imputador_num, normalizacion, discretizador, decodificador, caracteritisticas_seleccionadas, caracteritisticas_procesadas, 
                 grid, random_grid, validacion_grid, ensemble, reduccion_dimensionalidad)
    
    # entrenamientoNoSupervisado.main(random_state=random_state)
    
    
if __name__ == "__main__":
    main()