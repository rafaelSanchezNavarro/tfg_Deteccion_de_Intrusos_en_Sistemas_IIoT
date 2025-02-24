import os
import joblib
from scripts.explicabilidad import explicabilidad
from scripts.test import test

path = "DecisionTreeClassifier_0.9944"
model = os.path.join("modelos", path, f"{path}.pkl")
model_class2 = os.path.join("modelos", path, f"CalibratedClassifierCV_class2.pkl")
model_class1 = os.path.join("modelos", path, f"CalibratedClassifierCV_class1.pkl")

model = joblib.load(model)
model_class2 = joblib.load(model_class2)
model_class1 = joblib.load(model_class1)

test.main(model, path, model_class2, model_class1)
explicabilidad.main(model, path)