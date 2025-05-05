import os
import joblib
from scripts.anomalias import anomalias
from scripts.explicabilidad import explicabilidad
from scripts.test import test_opcion1

path = "LogisticRegression_0.9645"
model = os.path.join("modelos", path, f"LogisticRegression_0.9645.pkl")
model = joblib.load(model)

model_class2 = os.path.join("modelos", path, f"LogisticRegression_class2.pkl")
model_class2 = joblib.load(model_class2)

models_class1 = {}
for categoria in list(anomalias.keys()):
    try:
        model_class1 = os.path.join("modelos", path, f"{categoria}_class1.pkl")
        model_class1 = joblib.load(model_class1)
        models_class1[categoria] = model_class1
    except:
        pass

test_opcion1.main(model, path, model_class2, models_class1)
explicabilidad.main(model, path)

