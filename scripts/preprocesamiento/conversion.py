import pandas as pd
import ipaddress

def fix_dytype(df, umbral_numerico=0.7):
    object_cols = df.select_dtypes(include=['object']).columns
    int_cols = df.select_dtypes(include=['int64']).columns

    for col in object_cols:
        unique_values = set(df[col].unique())
        if unique_values.issubset({'true', 'false'}):
            df[col] = df[col].map({'true': True, 'false': False})
        elif len(unique_values) == 3 and 'true' in unique_values:
            # print(f"Columna {col} convertida a booleana, se han borrado {df[col].isna().sum()} filas que contenían 'nan'.")
            # df.dropna(subset=[col], inplace=True) # PREGUNTAR Los imputare mas tarde
            df[col] = df[col].map({'true': True, 'false': False})
        else:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > umbral_numerico:
                df[col] = converted.astype(float)
    
    for col in int_cols:
        if set(df[col].unique()).issubset({0, 1}):
            df[col] = df[col].astype(bool)
    
    return df
    
    # for col in object_cols:
    #   unique_values = set(df[col].unique())  # Valores únicos no nulos

    #   # Comprobar si los valores únicos representan booleanos
    #   boolean_like_values = {"true", "false", 'TRUE', 'FALSE'}
    #   if unique_values and len(unique_values) == 2 and boolean_like_values.issubset(unique_values):
    #         # Convertir a booleano: 'true' se convierte en True, 'false' en False
    #         df[col] = df[col].str.lower().map({'true': True, 'false': False, 'TRUE': True, 'FALSE': False})
    #         print(f"Columna {col} convertida a booleana.")
    #   else:
    #     # Intentar convertir otras columnas a numérico
    #     converted = pd.to_numeric(df[col], errors='coerce')
    #     # Verificar la proporción de valores numéricos válidos
    #     proportion_numeric = converted.notna().mean()

    #     if proportion_numeric > umbral_numerico:
    #         # Si la mayoría de los valores son numéricos, mantener como float
    #         df[col] = converted.astype(float)
    #         print(f"Columna {col} convertida a numérica.")
    #     else:
    #         # Si no cumple el criterio, no hacemos nada: mantiene los valores originales como string
    #         pass

    # # Convertir columnas enteras con valores únicos {0, 1} a booleanas
    # for col in int_cols:
    #     unique_values = df[col].unique()
    #     if set(unique_values).issubset({0, 1}):  # Verificar si los valores únicos son 0 y 1
    #         df[col] = df[col].astype(bool)
    #         print(f"2 Columna {col} convertida a booleana.")

    # return df

def clasificar_ip(ip):
    """Clasifica una IP como privada/pública y determina su clase."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        es_local = 1 if ip_obj.is_private else 0

        if isinstance(ip_obj, ipaddress.IPv4Address):
            primer_octeto = int(ip.split(".")[0])
            if 1 <= primer_octeto <= 126:
                clase = "A"
            elif 128 <= primer_octeto <= 191:
                clase = "B"
            elif 192 <= primer_octeto <= 223:
                clase = "C"
            elif 224 <= primer_octeto <= 239:
                clase = "D"
            elif 240 <= primer_octeto <= 255:
                clase = "E"
            else:
                clase = "Desconocida"
        else:
            clase = "IPv6"

        return es_local, clase
    except ValueError:
        return None, None

servicios_industriales = ["modbus", "mqtt", "coap"]

def tipo_servicio(series):
    """Crea una columna indicando si el servicio es industrial."""
    return series.apply(lambda x: 1 if x in servicios_industriales else 0)

def delete_ip_port(df):
    """Elimina las columnas 'ip' y 'port'."""
    return df.drop(columns=['Scr_IP', 'Scr_port', 'Des_IP', 'Des_port'])
