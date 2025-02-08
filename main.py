from datetime import datetime
import os
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from scripts.preprocesamiento import preprocesamiento
from scripts.preprocesamiento.preprocesamiento_utils import discretizers, scalers, imputers, encoders
from scripts.preprocesamiento.reduccion_dimensionalidad import seleccionar_variables_pca, seleccionar_variables_randomForest

from scripts.entrenamiento import entrenamiento
from scripts.entrenamiento.entrenamiento_utils.validacion import validation_methods
from modelos.diccionario_modelos import algorithms

from scripts.test import test

def guardar_conf(model, accuracy, imputador_cat=None, imputador_num=None, 
                 normalizacion=None, discretizador=None, decodificador=None, 
                 caracteristicas=None):
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

    if caracteristicas is not None:
        path_reduccion = os.path.join(output_dir, "caracteristicas.pkl")
        joblib.dump(caracteristicas, path_reduccion)
        
    # Guardar el modelo
    nombre_modelo = f"{model.__class__.__name__}_{accuracy:.4f}.pkl"
    path = os.path.join(output_dir, nombre_modelo)
    joblib.dump(model, path)
    print("Pipeline guardado exitosamente.\n")
    
    

def main():
    random_state = 42
    
    impuador_cat= imputers.imputers['categorical']['most_frequent']
    imputador_num = imputers.imputers['numeric']['mean']
    normalizacion = scalers.scalers['robust']
    discretizador = discretizers.discretizers['k_bins']
    decodificador = encoders.encoders['one_hot']
    reduccion_dimensionalidad = seleccionar_variables_pca
    
    # caracteristicas = preprocesamiento.main(
    #                     random_state,
    #                     impuador_cat, 
    #                     imputador_num, 
    #                     normalizacion, 
    #                     discretizador, 
    #                     decodificador, 
    #                     reduccion_dimensionalidad
    # )
    
    model = algorithms['DecisionTreeClassifier']()
    grid = True
    grid_n_iter = 1
    random_grid = True
    validacion_grid = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    
    model_train, accuracy = entrenamiento.main(
                        random_state,
                        model,
                        grid,
                        validacion_grid,
                        grid_n_iter,
                        random_grid
    )
    guardar_conf(model_train, accuracy)
    # guardar_conf(model_train, accuracy, impuador_cat, imputador_num, normalizacion, discretizador, decodificador, caracteristicas)
    
if __name__ == "__main__":
    main()
