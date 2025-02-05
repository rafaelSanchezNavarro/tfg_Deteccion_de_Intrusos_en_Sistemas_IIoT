from sklearn.impute import SimpleImputer

# Definir los imputadores
imputers = {
    'categorical': {
        'most_frequent': SimpleImputer(strategy='most_frequent'),
    },
    'numeric': {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median')
    }
}
