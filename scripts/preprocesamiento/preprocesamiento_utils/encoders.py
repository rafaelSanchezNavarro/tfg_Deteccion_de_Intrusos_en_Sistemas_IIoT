from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Definir codificadores
encoders = {
    "one_hot": OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
}
