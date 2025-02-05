import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from scripts.preprocesamiento.limpieza import replace_common_values, fix_mayus
from scripts.preprocesamiento.conversion import convert_numeric_columns, tipo_servicio, clasificar_ip
from scripts.preprocesamiento.calculos import calculo_varianza
from scripts.preprocesamiento.reduccion_dimensionalidad import correlacion_pares, correlacion_respecto_objetivo, seleccionar_variables_pca, seleccionar_variables_rfe, seleccionar_variables_randomForest
from scripts.preprocesamiento.preprocesamiento_utils import imputers, discretizers, scalers, encoders

random_state = 42  # Para reproducibilidad

def cargar_datos():
    """Carga el dataset original y lo devuelve como un DataFrame."""
    path = os.path.join("datos/raw", "X-IIoTID dataset.csv")  
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo.")
        return None

def dividir_datos(df):
    """Divide los datos en entrenamiento, validación y prueba."""
    X = df.drop(columns=['class1', 'class2', 'class3'])
    y_class3 = df['class3'].replace({'Normal': 0, 'Attack': 1})
    y_class2 = df['class2']
    y_class1 = df['class1']

    X_train, X_temp, y_train_class3, y_temp_class3, y_train_class2, y_temp_class2, y_train_class1, y_temp_class1 = train_test_split(
        X, y_class3, y_class2, y_class1, test_size=0.2, random_state=random_state, stratify=y_class3
    )

    X_val, X_test, y_val_class3, y_test_class3, y_val_class2, y_test_class2, y_val_class1, y_test_class1 = train_test_split(
        X_temp, y_temp_class3, y_temp_class2, y_temp_class1, test_size=0.5, random_state=random_state, stratify=y_temp_class3
    )

    return X_train, X_val, X_test, y_train_class3, y_val_class3, y_test_class3, y_train_class2, y_val_class2, y_test_class2, y_train_class1, y_val_class1, y_test_class1

def preprocesar_datos(X_train, X_val, y_train_class3):
    # Identificar columnas categóricas, numéricas y booleanas
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    boolean_cols = X_train.select_dtypes(include=['bool']).columns
    if boolean_cols.any():  # Si hay columnas booleanas
        X_train[boolean_cols] = X_train[boolean_cols].astype(int)
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    # Seleccionar el imputador deseado para características categóricas
    imputer_categorical = imputers.imputers['categorical']['most_frequent']
    X_train[categorical_cols] = imputer_categorical.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = imputer_categorical.transform(X_val[categorical_cols])
    
    # Seleccionar el imputador deseado para características numéricas
    imputer_numeric = imputers.imputers['numeric']['mean']
    X_train[numerical_cols] = imputer_numeric.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols] = imputer_numeric.transform(X_val[numerical_cols])

    ##############################################################################
    
    # Seleccionar el scaler deseado
    scaler = scalers.scalers['robust']
    # Ajustar el scaler con el set de entrenamiento y transformarlo
    # Se utiliza fit_transform en entrenamiento para calcular la media y desviación (o mediana y IQR en el caso de RobustScaler)
    X_train_scaled = scaler.fit_transform(X_train[numerical_cols])

    # Transformar el set de validación utilizando los parámetros calculados en el entrenamiento
    # Así se evita data leaking, ya que no se recalculan los parámetros con datos de validación
    X_val_scaled = scaler.transform(X_val[numerical_cols])

    # Convertir las matrices escaladas a DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=[f"{col}_scaled" for col in numerical_cols], index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=[f"{col}_scaled" for col in numerical_cols], index=X_val.index)

    ##############################################################################
    
    # Seleccionar el discretizador deseado
    discretizer = discretizers.discretizers['k_bins']  # Puedes cambiar a 'quantile_bins' si lo prefieres
    X_train_discrete = discretizer.fit_transform(X_train[numerical_cols])
    X_val_discrete = discretizer.transform(X_val[numerical_cols])

    # Convertir las matrices discretizadas a DataFrames
    X_train_discretized_df = pd.DataFrame(X_train_discrete, columns=[f"{col}_discrete" for col in numerical_cols], index=X_train.index)
    X_val_discretized_df = pd.DataFrame(X_val_discrete,  columns=[f"{col}_discrete" for col in numerical_cols], index=X_val.index)
    
    ##############################################################################
    
    processed_numeric_train = pd.concat([X_train_scaled_df, X_train_discretized_df], axis=1)
    processed_numeric_val = pd.concat([X_val_scaled_df, X_val_discretized_df], axis=1)
    
    ##############################################################################
    
    # Seleccionar el codificadores deseado para
    encoder = encoders.encoders['one_hot']
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_val_encoded = encoder.transform(X_val[categorical_cols])

    # Obtener los nombres de las nuevas columnas codificadas
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    # Convertir las matrices codificadas a DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_val_encoded_df = pd.DataFrame(X_val_encoded, columns=encoded_cols, index=X_val.index)
    
    ##############################################################################
    
    # Combinar con las características categóricas codificadas
    X_train_processed = pd.concat([processed_numeric_train, X_train_encoded_df], axis=1)
    X_val_processed = pd.concat([processed_numeric_val, X_val_encoded_df], axis=1)
    
    # Opcional: Reordenar las columnas si es necesario
    X_train_processed = X_train_processed.reindex(sorted(X_train_processed.columns), axis=1)
    X_val_processed = X_val_processed.reindex(sorted(X_val_processed.columns), axis=1)
    
    return X_train_processed, X_val_processed

def main():  
    print("\n🚀 Iniciando preprocesamiento...\n")
    
    df = cargar_datos()
    print(f"📊 Datos cargados: {df.shape}\n")
    
    (   
        X_train, X_val, X_test,
        y_train_class3, y_val_class3, y_test_class3,
        y_train_class2, y_val_class2, y_test_class2,
        y_train_class1, y_val_class1, y_test_class1
    ) = dividir_datos(df)
    
    print(f"📌 Datos divididos: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}\n")

    print("🔄 Preprocesando datos de entrenamiento...\n")
    X_train = convert_numeric_columns(X_train)
    X_train = replace_common_values(X_train)
    X_train = fix_mayus(X_train)
    print("✅ Preprocesamiento base completado para Train\n")
    
    print("🔄 Preprocesando datos de validación...\n")
    X_val = convert_numeric_columns(X_val)
    X_val = replace_common_values(X_val)
    X_val = fix_mayus(X_val)
    print("✅ Preprocesamiento base completado para Val\n")
    
    X_train['Instancia_completa'] = X_train.notnull().all(axis=1).astype(int)
    X_val['Instancia_completa'] = X_val.notnull().all(axis=1).astype(int)
    
    completas = X_train['Instancia_completa'].sum()
    incompletas = len(X_train) - completas
    print(f"ℹ️ Instancias completas: {completas}, incompletas: {incompletas}.\n")
    
    sample_weight_train = X_train['Instancia_completa'].replace({1: 3, 0: 1})
    
    columnas_no_comprobar = [col for col in df.columns if col not in ['Timestamp', 'Date'] and df[col].dtypes != 'object']
    variables_con_varianza_cero = calculo_varianza(X_train[columnas_no_comprobar])
    print(f"📉 Columnas con varianza cero eliminadas: {len(variables_con_varianza_cero)}\n")
    X_train = X_train.drop(columns=variables_con_varianza_cero)
    X_val = X_val.drop(columns=variables_con_varianza_cero)
    print(f"📏 Dimensiones después de eliminar varianza cero: Train {X_train.shape}, Val {X_val.shape}\n")
    
    X_train = X_train.drop(columns=['Timestamp', 'Date', 'Instancia_completa'], errors='ignore')
    X_val = X_val.drop(columns=['Timestamp', 'Date', 'Instancia_completa'], errors='ignore')
    
    alta_corr_pares = correlacion_pares(X_train, 0.97)
    print(f"🔗 Columnas con alta correlación eliminadas: {len(alta_corr_pares)}\n")
    X_train = X_train.drop(columns=alta_corr_pares)
    X_val = X_val.drop(columns=alta_corr_pares)
    
    baja_corr_respecto_obj = correlacion_respecto_objetivo(X_train, y_train_class3, 0.025)
    print(f"📉 Columnas con baja correlación respecto al objetivo eliminadas: {len(baja_corr_respecto_obj)}\n")
    X_train = X_train.drop(columns=baja_corr_respecto_obj)
    X_val = X_val.drop(columns=baja_corr_respecto_obj)
    print(f"📏 Dimensiones finales antes del procesamiento: Train {X_train.shape}, Val {X_val.shape}\n")
    
    print("🛠 Aplicando preprocesamiento final...\n")
    X_train_processed, X_val_processed = preprocesar_datos(X_train, X_val, y_train_class3)
    print(f"✅ Datos preprocesados: Train {X_train_processed.shape}, Val {X_val_processed.shape}\n")
    
    print("📊 Aplicando PCA para selección de variables...\n")
    X_train_processed, X_val_processed = seleccionar_variables_pca(X_train_processed, X_val_processed, n_components=0.95, num_top_features=10)
    print(f"✅ PCA aplicado: Train {X_train_processed.shape}, Val {X_val_processed.shape}\n")
    
    print(f"🎯 Preprocesamiento finalizado: {X_train_processed.shape} instancias de entrenamiento, {X_val_processed.shape} instancias de validación.\n")
    
    # Definir la ruta de la carpeta de salida
    output_dir = r"C:\Users\rafae\Desktop\tfg_Deteccion_de_Intrusos_en_Sistemas_IIoT\datos\procesados"

    # Crear la carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Guardar los conjuntos de entrenamiento y validación en la carpeta especificada
    X_train_processed.to_csv(os.path.join(output_dir, "X_train_processed.csv"), index=False)
    X_val_processed.to_csv(os.path.join(output_dir, "X_val_processed.csv"), index=False)

    # Guardar las etiquetas en la misma carpeta
    y_train_class3.to_csv(os.path.join(output_dir, "y_train_class3.csv"), index=False)
    y_val_class3.to_csv(os.path.join(output_dir, "y_val_class3.csv"), index=False)

    y_train_class2.to_csv(os.path.join(output_dir, "y_train_class2.csv"), index=False)
    y_val_class2.to_csv(os.path.join(output_dir, "y_val_class2.csv"), index=False)

    y_train_class1.to_csv(os.path.join(output_dir, "y_train_class1.csv"), index=False)
    y_val_class1.to_csv(os.path.join(output_dir, "y_val_class1.csv"), index=False)

    print(f"📁 Archivos guardados en: {output_dir}")

