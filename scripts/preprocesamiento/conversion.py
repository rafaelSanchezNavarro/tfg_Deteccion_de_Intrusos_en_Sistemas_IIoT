import pandas as pd
import ipaddress

def convert_numeric_columns(df, umbral_numerico=0.5):
    """Convierte columnas de texto a numéricas si tienen más del umbral de valores numéricos."""
    object_cols = df.select_dtypes(include=['object']).columns
    int_cols = df.select_dtypes(include=['int64']).columns

    for col in object_cols:
        unique_values = set(df[col].unique())
        if unique_values and len(unique_values) == 2 and unique_values.issubset({"true", "false", "TRUE", "FALSE"}):
            df[col] = df[col].str.lower().map({'true': True, 'false': False})
        else:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > umbral_numerico:
                df[col] = converted.astype(float)
    
    for col in int_cols:
        if set(df[col].unique()).issubset({0, 1}):
            df[col] = df[col].astype(bool)
    
    return df

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
