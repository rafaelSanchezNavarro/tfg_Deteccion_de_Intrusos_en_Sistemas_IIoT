import os
import joblib
from scripts.anomalias import anomalias
from scripts.explicabilidad import explicabilidad
from scripts.test import test_opcion3

path = "pruebas arquitecturas/DecisionTreeClassifier_0.9937 arq3"
model_class2 = os.path.join("modelos", path, f"DecisionTreeClassifier_0.9937.pkl")
model_class2 = joblib.load(model_class2)

models_class1 = {}
for categoria in list(anomalias.keys()):
    try:
        model_class1 = os.path.join("modelos", path, f"{categoria}_class1.pkl")
        model_class1 = joblib.load(model_class1)
        models_class1[categoria] = model_class1
    except:
        pass

test_opcion3.main(model_class2, path, models_class1)
explicabilidad.main(model_class2, path)