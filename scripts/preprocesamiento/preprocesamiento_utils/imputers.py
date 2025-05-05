from sklearn.impute import KNNImputer, SimpleImputer

# Definir los imputadores
imputers = {
    'categorical': {
        'most_frequent': SimpleImputer(strategy='most_frequent'),
        'knn': KNNImputer(n_neighbors=5),
        'constant': SimpleImputer(strategy='constant', fill_value='missing')
    },
    'numeric': {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median')
    }
}
