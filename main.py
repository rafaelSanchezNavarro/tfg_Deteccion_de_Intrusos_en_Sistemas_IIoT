from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from scripts.preprocesamiento import preprocesamiento
from scripts.preprocesamiento.preprocesamiento_utils import discretizers, scalers, imputers, encoders
from scripts.preprocesamiento.reduccion_dimensionalidad import seleccionar_variables_pca

from scripts.entrenamiento import entrenamiento
from scripts.entrenamiento.entrenamiento_utils.validacion import validation_methods
from modelos.diccionario_modelos import algorithms

def main():
    random_state = 42
    
    # impuador_cat= imputers.imputers['categorical']['most_frequent']
    # imputador_num = imputers.imputers['numeric']['mean']
    # normalizacion = scalers.scalers['robust']
    # discretizador = discretizers.discretizers['k_bins']
    # decodificador = encoders.encoders['one_hot']
    # reduccion_dimensionalidad = seleccionar_variables_pca
    
    # preprocesamiento.main(
    #                     random_state,
    #                     impuador_cat, 
    #                     imputador_num, 
    #                     normalizacion, 
    #                     discretizador, 
    #                     decodificador, 
    #                     reduccion_dimensionalidad
    # )
    
    model = algorithms['DecisionTreeClassifier'](random_state=random_state)
    grid = True
    grid_n_iter = 2
    random_grid = False
    validacion_grid = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=random_state)

    entrenamiento.main(
                        random_state,
                        model,
                        grid,
                        validacion_grid,
                        grid_n_iter,
                        random_grid
    )
        

if __name__ == "__main__":
    main()
