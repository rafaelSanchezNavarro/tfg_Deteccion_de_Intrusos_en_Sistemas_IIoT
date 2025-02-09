from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler

# Definir escaladores
scalers = {
    "robust": RobustScaler(),
    "standard": StandardScaler()
}
