import os
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from scripts.entrenamiento.entrenamiento_utils.grid import param_grid
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize
from modelos.diccionario_modelos import algorithms

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
    path_X_val = os.path.join(carpeta, "X_val.csv")
    try:
        datos["X_val"] = pd.read_csv(path_X_val, low_memory=False)
        print(f"✅ X_val cargado: {datos['X_val'].shape[0]} filas, {datos['X_val'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_val}.")
        return None

    # Cargar y_train_class3
    path_y_train_class3 = os.path.join(carpeta, "y_train_class3.csv")
    try:
        datos["y_train_class3"] = pd.read_csv(path_y_train_class3, low_memory=False)
        print(f"✅ y_train_class3 cargado: {datos['y_train_class3'].shape[0]} filas, {datos['y_train_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class3}.")
        return None

    # Cargar y_val_class3
    path_y_val_class3 = os.path.join(carpeta, "y_val_class3.csv")
    try:
        datos["y_val_class3"] = pd.read_csv(path_y_val_class3, low_memory=False)
        print(f"✅ y_val_class3 cargado: {datos['y_val_class3'].shape[0]} filas, {datos['y_val_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class3}.")
        return None

    # Cargar y_train_class2
    path_y_train_class2 = os.path.join(carpeta, "y_train_class2.csv")
    try:
        datos["y_train_class2"] = pd.read_csv(path_y_train_class2, low_memory=False)
        print(f"✅ y_train_class2 cargado: {datos['y_train_class2'].shape[0]} filas, {datos['y_train_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class2}.")
        return None

    # Cargar y_val_class2
    path_y_val_class2 = os.path.join(carpeta, "y_val_class2.csv")
    try:
        datos["y_val_class2"] = pd.read_csv(path_y_val_class2, low_memory=False)
        print(f"✅ y_val_class2 cargado: {datos['y_val_class2'].shape[0]} filas, {datos['y_val_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class2}.")
        return None

    # Cargar y_train_class1
    path_y_train_class1 = os.path.join(carpeta, "y_train_class1.csv")
    try:
        datos["y_train_class1"] = pd.read_csv(path_y_train_class1, low_memory=False)
        print(f"✅ y_train_class1 cargado: {datos['y_train_class1'].shape[0]} filas, {datos['y_train_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class1}.")
        return None

    # Cargar y_val_class1
    path_y_val_class1 = os.path.join(carpeta, "y_val_class1.csv")
    try:
        datos["y_val_class1"] = pd.read_csv(path_y_val_class1, low_memory=False)
        print(f"✅ y_val_class1 cargado: {datos['y_val_class1'].shape[0]} filas, {datos['y_val_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class1}.")
        return None

    return datos

def clasificacion_binaria(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3, ensemble):

        y_train_class3 = y_train_class3.values.ravel()
        y_val_class3 = y_val_class3.values.ravel()
        
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train[boolean_cols] = X_train[boolean_cols].astype(float)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        if ensemble and grid:
            tree_model = model.estimators[0][1]
        
        if grid:
            # X_train_sampled = X_train.sample(n=10000, random_state=random_state)
            # y_train_class3_sampled  = y_train_class3.loc[X_train_sampled.index]
            
            # X_train = X_train.drop(index=X_train_sampled.index)
            # y_train_class3 = y_train_class3.drop(index=X_train_sampled.index)
            
            if ensemble:
                grid_model = tree_model
            else:
                grid_model = model
                
            grid_search = optimize(
                random_grid=random_grid,
                random_state=random_state,
                estimator=grid_model,
                X=X_train,
                y=y_train_class3,
                param_grid=param_grid[grid_model.__class__.__name__],
                n_iter=grid_n_iter,
                cv=validacion_grid,
                scoring='accuracy',
                n_jobs=-1,
            )
            
            grid_model = grid_search[0].best_estimator_
            print(f"➡️  Optimización completa para {grid_model.__class__.__name__}.")

            if ensemble:
                model.estimators[0] = ('mwbp', grid_model)
            
        print("➡️  Creando el pipeline...")
        pipeline = create_pipeline(
            model=model,  # Modelo del algoritmo final (ensemble)
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("➡️  Pipeline creado exitosamente.")
        
        # Validación cruzada de 5 pliegues
        print("➡️  Realizando validación cruzada de 5 pliegues...")
        cv_scores = cross_val_score(pipeline, X_train, y_train_class3, cv=5, scoring='accuracy')
        print("📈 CV scores:", cv_scores)
        print("📈 Accuracy media (CV): {:.4f}".format(cv_scores.mean()))
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        print("➡️  Entrenando el pipeline...")
        pipeline.fit(X_train, y_train_class3)
        print("➡️  Entrenamiento completo.")


        # Realizar predicciones
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class3 = pipeline.predict(X_val)
        print("➡️  Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class3, y_pred_class3)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
        
        precision = precision_score(y_val_class3, y_pred_class3)
        print(f'📈 Precision (validacion): {precision:.4f}')
        
        recall = recall_score(y_val_class3, y_pred_class3)
        print(f'📈 Recall (validacion): {recall:.4f}')
        
        f1 = f1_score(y_val_class3, y_pred_class3)
        print(f'📈 F1 (validacion): {f1:.4f}')
        
        roc = roc_auc_score(y_val_class3, y_pred_class3)
        print(f'📈 ROC (validacion): {roc:.4f}')
        
        return pipeline, accuracy, precision, recall, f1, roc

def clasificacion_multiclase_categoria(random_state, X_train, X_val, y_train_class3, y_val_class3, y_train_class2 , y_val_class2):
        indices_train = np.where(y_train_class3.values == 1)[0]
        X_train_class2 = X_train.iloc[indices_train]
        y_train_class2_filtered = y_train_class2.iloc[indices_train]

        indices_val = np.where(y_val_class3.values == 1)[0]
        X_val_class2 = X_val.iloc[indices_val]
        y_val_class2_filtered = y_val_class2.iloc[indices_val]
        
        y_train_class2_filtered = y_train_class2_filtered.values.ravel()
        y_val_class2_filtered = y_val_class2_filtered.values.ravel()
        
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train_class2.select_dtypes(include=['object']).columns
        boolean_cols = X_train_class2.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train_class2[boolean_cols] = X_train_class2[boolean_cols].astype(int)
        numerical_cols = X_train_class2.select_dtypes(include=['float64', 'int64']).columns
        
        model = algorithms['DecisionTreeClassifier'](random_state=random_state)
        print("➡️  Creando el pipeline multiclase categoria...")
        pipeline = create_pipeline(
            model=model,  # Modelo del algoritmo final (ensemble)
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("➡️  Pipeline multiclase categoria creado exitosamente.")
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        print("➡️  Entrenando el pipeline multiclase categoria...")
        pipeline.fit(X_train_class2, y_train_class2_filtered)
        print("➡️  Entrenamiento completo.")


        # Realizar predicciones
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class2 = pipeline.predict(X_val_class2)
        print("➡️  Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class2_filtered, y_pred_class2)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')

def clasificacion_multiclase_tipo(random_state, X_train, X_val, y_train_class3, y_val_class3, y_train_class1 , y_val_class1):
        indices_train = np.where(y_train_class3.values == 1)[0]
        X_train_class1 = X_train.iloc[indices_train]
        y_train_class1_filtered = y_train_class1.iloc[indices_train]

        indices_val = np.where(y_val_class3.values == 1)[0]
        X_val_class1 = X_val.iloc[indices_val]
        y_val_class1_filtered = y_val_class1.iloc[indices_val]
        
        y_train_class1_filtered = y_train_class1_filtered.values.ravel()
        y_val_class1_filtered = y_val_class1_filtered.values.ravel()
        
        # Identificar columnas categóricas, numéricas y booleanas
        categorical_cols = X_train_class1.select_dtypes(include=['object']).columns
        boolean_cols = X_train_class1.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  # Si hay columnas booleanas
            X_train_class1[boolean_cols] = X_train_class1[boolean_cols].astype(int)
        numerical_cols = X_train_class1.select_dtypes(include=['float64', 'int64']).columns
        
        model = algorithms['DecisionTreeClassifier'](random_state=random_state)
        print("➡️  Creando el pipeline multiclase tipo...")
        pipeline = create_pipeline(
            model=model,  # Modelo del algoritmo final (ensemble)
            categorical_features=categorical_cols,  # Columnas categóricas
            numerical_features=numerical_cols,  # Columnas numéricas
        )
        print("➡️  Pipeline multiclase tipo creado exitosamente.")
        
        # Entrenar el pipeline completo (incluyendo preprocesamiento y RFE)
        print("➡️  Entrenando el pipeline multiclase tipo...")
        pipeline.fit(X_train_class1, y_train_class1_filtered)
        print("➡️  Entrenamiento completo.")


        # Realizar predicciones
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class1 = pipeline.predict(X_val_class1)
        print("➡️  Predicciones realizadas.")


        # Evaluar el rendimiento
        accuracy = accuracy_score(y_val_class1_filtered, y_pred_class1)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
        
        # print(classification_report(y_val_class1_filtered, y_pred_class1))
        
def main(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, ensemble):  
    print("🚀 Iniciando entrenamiento...")
    
    # Cargar todos los archivos de datos procesados
    datos = cargar_datos()

    # Asignar los DataFrames a variables individuales
    X_train = datos["X_train"]
    X_val = datos["X_val"]

    y_train_class3 = datos["y_train_class3"]
    y_val_class3 = datos["y_val_class3"]
    
    y_train_class2 = datos["y_train_class2"]
    y_val_class2 = datos["y_val_class2"]
    
    y_train_class1 = datos["y_train_class1"]
    y_val_class1 = datos["y_val_class1"]
    
    # Entrenar el modelo
    pipeline, accuracy, precision, recall, f1, roc = clasificacion_binaria(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class3, y_val_class3, ensemble)
    clasificacion_multiclase_categoria(random_state, X_train, X_val, y_train_class3, y_val_class3, y_train_class2 , y_val_class2)
    clasificacion_multiclase_tipo(random_state, X_train, X_val, y_train_class3, y_val_class3, y_train_class1 , y_val_class1)
    
    print("🎯 Entrenamiento finalizado")
    pipeline = pipeline.named_steps['model']
    return pipeline, accuracy, precision, recall, f1, roc
