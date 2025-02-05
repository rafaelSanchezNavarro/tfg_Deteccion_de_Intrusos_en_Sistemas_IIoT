import pandas as pd
import numpy as np

# Reemplazos comunes de valores
common_replacements = {
    '-': np.nan,
    '?': np.nan,
    'nan': np.nan,
}

def replace_common_values(df):
    """Reemplaza valores comunes como '-', '?' y 'nan' por NaN."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace(common_replacements)
    return df

def fix_mayus(df):
    """Convierte todas las cadenas de texto a minúsculas y convierte 'true'/'false' a booleanos."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
        df[col] = df[col].replace({'true': True, 'false': False})
    return df
