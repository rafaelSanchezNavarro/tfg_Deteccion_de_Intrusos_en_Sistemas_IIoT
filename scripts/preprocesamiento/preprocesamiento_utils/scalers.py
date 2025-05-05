from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Definir escaladores
scalers = {
    "robust": RobustScaler(),
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
}
