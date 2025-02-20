from scipy.stats import zscore
import numpy as np

# def outliers(df):
#     df_numeric = df.select_dtypes(include=['float', 'int'])  # Seleccionar solo columnas numéricas

#     threshold = 3  # Umbral de Z-score
#     z_scores = np.abs(zscore(df_numeric))  # Cálculo del Z-score

#     mask = (z_scores < threshold).all(axis=1)  # Filtrar filas sin outliers

#     df_clean = df[mask]  # Conservar filas sin outliers, manteniendo todas las columnas

#     print(f"Outliers eliminados con Z-score: {len(df) - len(df_clean)}")

#     return df_clean  # Retornar el DataFrame limpio
    
       
def outliers(df):
    df_numeric = df.select_dtypes(include=['float', 'int'])  # Seleccionar solo columnas numéricas

    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Crear una máscara para eliminar filas con al menos un outlier
    mask = ~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)

    df_clean = df[mask]  # Conservar solo filas sin outliers

    print(f"Outliers eliminados con IQR: {len(df) - len(df_clean)}")

    return df_clean  # Retorna el DataFrame sin outliers

