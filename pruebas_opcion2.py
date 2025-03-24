import os
import joblib
from scripts.explicabilidad import explicabilidad
from scripts.test import test_opcion2

path = "DecisionTreeClassifier_0.9944"
model = os.path.join("modelos", path, f"{path}.pkl")
model = joblib.load(model)

model_class2 = os.path.join("modelos", path, f"DecisionTreeClassifier_class2.pkl")
model_class2 = joblib.load(model_class2)

categorias = ["RDOS"]
models_class1 = {}
for categoria in categorias:
    model_class1 = os.path.join("modelos", path, f"DecisionTreeClassifier_class1.pkl")
    model_class1 = joblib.load(model_class1)
    models_class1[categoria] = model_class1

test_opcion2.main(model, path, model_class2, models_class1)
explicabilidad.main(model, path)