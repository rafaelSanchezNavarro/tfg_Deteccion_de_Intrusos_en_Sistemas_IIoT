import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def optimize(random_grid, estimator, X, y, param_grid, random_state=None, n_iter=None, scoring=None, cv=None, n_jobs=None, refit=True,
             verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False):

  """
    estimator:
        El estimador (modelo) que se quiere optimizar.

    X:
        Datos de entrada.

    y:
        Etiquetas de salida.

    param_grid:
        Diccionario con los hiperparámetros a probar. **Este parámetro es obligatorio**.

    scoring:
        Métrica de evaluación. **Valor por defecto**: `None`, lo que significa que se usará la métrica por defecto:
        - Para clasificación: `accuracy`.
        - Para regresión: `r2`.
        También puede ser un diccionario para evaluar múltiples métricas.

    cv:
        Número de particiones para la validación cruzada. **Valor por defecto**: `None`, lo que significa que se usará 5 particiones (`cv=5`).

    n_jobs:
        Número de trabajos paralelos para la búsqueda. **Valor por defecto**: `None` (1 núcleo).
        - `-1` para usar todos los núcleos disponibles.

    refit:
        Si es `True`, ajusta el modelo con los mejores parámetros encontrados.
        También puede ser el nombre de una métrica si se utiliza scoring múltiple.
        **Valor por defecto**: `True`.

    verbose:
        Nivel de detalles de los mensajes. **Valor por defecto**: `0` (sin salida).

    pre_dispatch:
        Número de trabajos a despachar antes de ejecutar en paralelo.
        **Valor por defecto**: `'2*n_jobs'`.

    error_score:
        Puntuación a asignar si ocurre un error durante el ajuste. **Valor por defecto**: `np.nan`.
        - `'raise'` lanza una excepción si hay errores.

    return_train_score:
        Si `True`, se devolverán los puntajes del conjunto de entrenamiento en los resultados.
        **Valor por defecto**: `False`.
  """

  best_metric = None
  all_metrics = {
      'accuracy' : None,
      'precision': None,
      'recall': None,
      'f1': None,
  }
  best_metric_score = 0
  metrics = ('accuracy', 'precision', 'recall', 'f1')

  if not random_grid: # False
      if scoring is not None: # Viene
          model = GridSearchCV(estimator=estimator,
                                param_grid=param_grid,
                                scoring=scoring,
                                cv=cv,
                                n_jobs=n_jobs,
                                refit=refit,
                                verbose=verbose,
                                pre_dispatch=pre_dispatch,
                                error_score=error_score,
                                return_train_score=return_train_score)
          classifier = model.fit(X, y)
          all_metrics[scoring] = model.best_score_
          best_metric = scoring
      else:
          for metric in metrics:
              model = GridSearchCV(estimator=estimator,
                                    param_grid=param_grid,
                                    scoring=metric,
                                    cv=cv,
                                    n_jobs=n_jobs,
                                    refit=refit,
                                    verbose=verbose,
                                    pre_dispatch=pre_dispatch,
                                    error_score=error_score,
                                    return_train_score=return_train_score)
              classifier = model.fit(X, y)
              all_metrics[metric] = model.best_score_
              if best_metric_score < model.best_score_:
                  best_metric_score = model.best_score_
                  best_metric = metric

  else: # True
      if scoring is not None: # Viene
          model = RandomizedSearchCV(estimator=estimator,
                                      param_distributions=param_grid,
                                      random_state=random_state,
                                      scoring=scoring,
                                      n_iter=n_iter,
                                      cv=cv,
                                      n_jobs=n_jobs,
                                      refit=refit,
                                      verbose=verbose,
                                      pre_dispatch=pre_dispatch,
                                      error_score=error_score,
                                      return_train_score=return_train_score)
          classifier = model.fit(X, y)
          all_metrics[scoring] = model.best_score_
          best_metric = scoring
      else:
          for metric in metrics:
              model = RandomizedSearchCV(estimator=estimator,
                                          param_distributions=param_grid,
                                          scoring=metric,
                                          n_iter=n_iter,
                                          cv=cv,
                                          n_jobs=n_jobs,
                                          refit=refit,
                                          verbose=verbose,
                                          pre_dispatch=pre_dispatch,
                                          error_score=error_score,
                                          return_train_score=return_train_score)
              classifier = model.fit(X, y)
              all_metrics[metric] = model.best_score_
              if best_metric_score < model.best_score_:
                  best_metric_score = model.best_score_
                  best_metric = metric

  return classifier, best_metric, all_metrics
