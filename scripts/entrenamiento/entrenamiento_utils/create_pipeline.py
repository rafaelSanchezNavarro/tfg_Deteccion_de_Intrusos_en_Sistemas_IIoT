import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def create_pipeline(model=None,
                   imputer_categorical=None,
                   imputer_numeric=None,
                   scaler=None,
                   discretizer=None,
                   encoders=None,
                   categorical_features=None,
                   numerical_features=None):
    """
    Crea un pipeline de preprocesamiento y modelo.

    Parámetros:
    - model: Un objeto de modelo de sklearn que tiene métodos `fit` y `predict`.
    - imputer_categorical: Instancia de imputador para características categóricas.
    - imputer_numeric: Instancia de imputador para características numéricas.
    - discretizer: Instancia de discretizador para características numéricas.
    - encoder: Instancia de codificador para características categóricas.
    - categorical_features: Lista de nombres de características categóricas.
    - numerical_features: Lista de nombres de características numéricas.
    - feature_selection: Objeto de selección de características (e.g., RFE).
    """

    # Asegurarse de que categorical_features y numerical_features sean listas
    if categorical_features is not None and isinstance(categorical_features, pd.Index):
        categorical_features = categorical_features.tolist()

    if numerical_features is not None and isinstance(numerical_features, pd.Index):
        numerical_features = numerical_features.tolist()

    # Pipeline para características numéricas
    numeric_pipeline_steps = []
    if imputer_numeric:
        numeric_pipeline_steps.append(('imputer', imputer_numeric))
    if scaler:
        numeric_pipeline_steps.append(('scaler', scaler))
    if discretizer:
        numeric_pipeline_steps.append(('discretizer', discretizer))

    numeric_pipeline = Pipeline(numeric_pipeline_steps) if numeric_pipeline_steps else 'passthrough'

    # Pipeline para características categóricas
    categorical_pipeline_steps = []
    if imputer_categorical:
        categorical_pipeline_steps.append(('imputer', imputer_categorical))
    if encoders:
        categorical_pipeline_steps.append(('encoder', encoders))

    categorical_pipeline = Pipeline(categorical_pipeline_steps) if categorical_pipeline_steps else 'passthrough'

    # Crear el ColumnTransformer con pipelines secuenciales
    if numerical_features and categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=True)
    else:
        preprocessor = 'passthrough'

    # Crear el pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline