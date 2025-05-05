import os
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.utils import compute_class_weight
from scripts.anomalias import anomalias
from scripts.entrenamiento.entrenamiento_utils.balanceo_pesos import obtener_pesos_suavizados
from scripts.entrenamiento.entrenamiento_utils.grid import param_grid
from scripts.entrenamiento.entrenamiento_utils.create_pipeline import create_pipeline
from scripts.entrenamiento.entrenamiento_utils.optimize import optimize
from modelos.diccionario_modelos import algorithms

def cargar_datos():

    carpeta = r"datos/preprocesados"
    datos = {}

    path_X_train = os.path.join(carpeta, "X_train.csv")
    try:
        datos["X_train"] = pd.read_csv(path_X_train, low_memory=False)
        print(f"✅ X_train cargado: {datos['X_train'].shape[0]} filas, {datos['X_train'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_train}.")
        return None

    path_X_val = os.path.join(carpeta, "X_val.csv")
    try:
        datos["X_val"] = pd.read_csv(path_X_val, low_memory=False)
        print(f"✅ X_val cargado: {datos['X_val'].shape[0]} filas, {datos['X_val'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_X_val}.")
        return None

    path_y_train_class3 = os.path.join(carpeta, "y_train_class3.csv")
    try:
        datos["y_train_class3"] = pd.read_csv(path_y_train_class3, low_memory=False)
        print(f"✅ y_train_class3 cargado: {datos['y_train_class3'].shape[0]} filas, {datos['y_train_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class3}.")
        return None

    path_y_val_class3 = os.path.join(carpeta, "y_val_class3.csv")
    try:
        datos["y_val_class3"] = pd.read_csv(path_y_val_class3, low_memory=False)
        print(f"✅ y_val_class3 cargado: {datos['y_val_class3'].shape[0]} filas, {datos['y_val_class3'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class3}.")
        return None

    path_y_train_class2 = os.path.join(carpeta, "y_train_class2.csv")
    try:
        datos["y_train_class2"] = pd.read_csv(path_y_train_class2, low_memory=False)
        print(f"✅ y_train_class2 cargado: {datos['y_train_class2'].shape[0]} filas, {datos['y_train_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class2}.")
        return None

    path_y_val_class2 = os.path.join(carpeta, "y_val_class2.csv")
    try:
        datos["y_val_class2"] = pd.read_csv(path_y_val_class2, low_memory=False)
        print(f"✅ y_val_class2 cargado: {datos['y_val_class2'].shape[0]} filas, {datos['y_val_class2'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class2}.")
        return None

    path_y_train_class1 = os.path.join(carpeta, "y_train_class1.csv")
    try:
        datos["y_train_class1"] = pd.read_csv(path_y_train_class1, low_memory=False)
        print(f"✅ y_train_class1 cargado: {datos['y_train_class1'].shape[0]} filas, {datos['y_train_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_train_class1}.")
        return None

    path_y_val_class1 = os.path.join(carpeta, "y_val_class1.csv")
    try:
        datos["y_val_class1"] = pd.read_csv(path_y_val_class1, low_memory=False)
        print(f"✅ y_val_class1 cargado: {datos['y_val_class1'].shape[0]} filas, {datos['y_val_class1'].shape[1]} columnas.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {path_y_val_class1}.")
        return None

    return datos

def clasificacion_multiclase_categoria(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class2 , y_val_class2, ensemble):
        
        
        y_train_class2 = y_train_class2.values.ravel()
        y_val_class2 = y_val_class2.values.ravel()

        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  
            X_train[boolean_cols] = X_train[boolean_cols].astype(float)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        if ensemble and grid:
            tree_model = model.estimators[0][1]
        
        if grid:
            if ensemble:
                grid_model = tree_model
            else:
                grid_model = model
                
            grid_search = optimize(
                random_grid=random_grid,
                random_state=random_state,
                estimator=grid_model,
                X=X_train,
                y=y_train_class2,
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
            else:
                model = grid_model
                
        pesos = compute_class_weight('balanced', classes=np.array(list(anomalias.keys()) + ["Normal"]), y=y_train_class2)
        pesos_categorias = dict(zip(list(anomalias.keys()), pesos))
        # pesos_categorias = obtener_pesos_suavizados(y_train_class2)
        
        print("➡️  Creando el pipeline multiclase tipo...")
        model.set_params(class_weight=pesos_categorias)
        pipeline = create_pipeline(
            model=model,  
            categorical_features=categorical_cols,  
            numerical_features=numerical_cols,  
        )
        print("➡️  Pipeline multiclase tipo creado exitosamente.")
        
        print("➡️  Entrenando el pipeline multiclase tipo...")
        pipeline.fit(X_train, y_train_class2)
        print("➡️  Entrenamiento completo.")
        print("➡️  Realizando predicciones en el conjunto de validación...")
        y_pred_class2 = pipeline.predict(X_val)
        print("➡️  Predicciones realizadas.")
        
        accuracy = accuracy_score(y_val_class2, y_pred_class2)
        print(f'📈 Accuracy (validacion): {accuracy:.4f}')
        precision = precision_score(y_val_class2, y_pred_class2, average='macro', zero_division=0)
        print(f'📈 Precision (validacion): {precision:.4f}')
        recall = recall_score(y_val_class2, y_pred_class2, average='macro')
        print(f'📈 Recall (validacion): {recall:.4f}')
        f1 = f1_score(y_val_class2, y_pred_class2, average='macro')
        print(f'📈 F1 (validacion): {f1:.4f}')
        y_pred_proba = pipeline.predict_proba(X_val)  
        roc = roc_auc_score(y_val_class2, y_pred_proba, multi_class='ovr', average='macro')
        
        return pipeline, y_pred_class2, accuracy, precision, recall, f1, roc

def clasificacion_multiclase_tipo(model, X_train, X_val, y_train_class1 , y_val_class1, y_pred_class2, y_train_class2 , y_val_class2):
        
        pipeline_tipos = {}
        
        indices_train = np.where(y_train_class2.values != "Normal")[0]
        X_train = X_train.iloc[indices_train]
        y_train_class1 = y_train_class1.iloc[indices_train].values.ravel()
        y_train_class2 = y_train_class2.iloc[indices_train].values.ravel()
        
        indices_val = np.where(y_pred_class2 != "Normal")[0]
        X_val = X_val.iloc[indices_val]
        y_val_class1 = y_val_class1.iloc[indices_val].values.ravel()
        y_val_class2 = y_val_class2.iloc[indices_val].values.ravel()
            
        tipos = [tipo for tipos_lista in anomalias.values() for tipo in tipos_lista]

        pesos = compute_class_weight('balanced', classes= np.array(tipos), y=y_train_class1)
        pesos_tipos = dict(zip(tipos, pesos))
        # pesos_tipos = obtener_pesos_suavizados(y_train_class1)
        
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        boolean_cols = X_train.select_dtypes(include=['bool']).columns
        if boolean_cols.any():  
            X_train[boolean_cols] = X_train[boolean_cols].astype(int)
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        indices_anomalias = np.where(y_pred_class2 != "Normal")[0]
        y_pred_class2 = y_pred_class2[indices_anomalias]
        
        categorias_multiples_tipos = [key for key, value in anomalias.items() if len(value) > 1]
        
        for categoria in list(anomalias.keys()):

            indices_train_categoria = np.where(y_train_class2 == categoria)[0]
            X_train_categoria = X_train.iloc[indices_train_categoria]
            y_train_class1_categoria = y_train_class1[indices_train_categoria]
            
            if categoria not in categorias_multiples_tipos:
                continue 
            
            indices_anomalia_prediccion_categoria = np.where(y_pred_class2 == categoria)[0]           
            X_val_tipos = X_val.iloc[indices_anomalia_prediccion_categoria]
            y_val_class1_tipos = y_val_class1[indices_anomalia_prediccion_categoria]

            modelo_categoria = clone(model)
            modelo_categoria.set_params(class_weight=pesos_tipos)
            pipeline = create_pipeline(
                model=modelo_categoria,
                categorical_features=categorical_cols,  
                numerical_features=numerical_cols,  
            )
            
            print(f"➡️  Entrenando el pipeline multiclase tipo: {categoria}...")
            pipeline.fit(X_train_categoria, y_train_class1_categoria)
            print("➡️  Realizando predicciones en el conjunto de validación...")
            y_pred_class1 = pipeline.predict(X_val_tipos)
            print("➡️  Predicciones realizadas.")

            indices_anomalias_reales = np.where(np.isin(y_val_class1_tipos, anomalias.get(categoria, [])))[0]
            y_val_class1_categoria_reales = y_val_class1_tipos[indices_anomalias_reales]
            y_pred_class1_tipos_reales = y_pred_class1[indices_anomalias_reales]
            
            accuracy = accuracy_score(y_val_class1_categoria_reales, y_pred_class1_tipos_reales)
            print(f'📈 Accuracy (validacion): {accuracy:.4f}')
            precision = precision_score(y_val_class1_categoria_reales, y_pred_class1_tipos_reales, average='macro', zero_division=0)
            print(f'📈 Precision (validacion): {precision:.4f}')
            recall = recall_score(y_val_class1_categoria_reales, y_pred_class1_tipos_reales, average='macro')
            print(f'📈 Recall (validacion): {recall:.4f}')
            f1 = f1_score(y_val_class1_categoria_reales, y_pred_class1_tipos_reales, average='macro')
            print(f'📈 F1 (validacion): {f1:.4f}')
        
            pipeline_tipos[categoria] = pipeline
            
        return pipeline_tipos
        
def main(random_state, model, grid, validacion_grid, grid_n_iter, random_grid, ensemble, model_class2, model_class1):  
    print("🚀 Iniciando entrenamiento...")
    
    datos = cargar_datos()

    X_train = datos["X_train"]
    X_val = datos["X_val"]
    
    y_train_class2 = datos["y_train_class2"]
    y_val_class2 = datos["y_val_class2"]
    
    y_train_class1 = datos["y_train_class1"]
    y_val_class1 = datos["y_val_class1"]
    
    pipeline_class2, y_pred_class2, accuracy, precision, recall, f1, roc = clasificacion_multiclase_categoria(random_state, model_class2, grid, validacion_grid, grid_n_iter, random_grid, X_train, X_val, y_train_class2 , y_val_class2, ensemble)
    pipelines_class1 = clasificacion_multiclase_tipo(model_class1, X_train, X_val, y_train_class1 , y_val_class1, y_pred_class2, y_train_class2 , y_val_class2)
    
    print("🎯 Entrenamiento finalizado")
    
    pipeline_class2 = pipeline_class2.named_steps['model']
    
    return pipeline_class2, accuracy, precision, recall, f1, roc, None, pipelines_class1
