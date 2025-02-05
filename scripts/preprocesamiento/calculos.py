
def calculo_varianza(df):
    """Calcula las varianzas de las columnas de un DataFrame y devuelve las que tienen varianza igual a cero."""
    varianzas = df.var()

    # Identificar columnas con varianza igual a cero
    variables_con_varianza_cero = [col for col, varianza in varianzas.items() if varianza == 0]
    
    return variables_con_varianza_cero

