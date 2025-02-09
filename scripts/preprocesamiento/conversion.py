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
            print(f"Columna {col} convertida a booleana, se han borrado {df[col].isna().sum()} filas que contenían 'nan'.")
            df.dropna(subset=[col], inplace=True)
            df[col] = df[col].map({'true': True, 'false': False})
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

def delete_ip_port(df):
    """Elimina las columnas 'ip' y 'port'."""
    return df.drop(columns=['Scr_IP', 'Scr_port', 'Des_IP', 'Des_port'])
