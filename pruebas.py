import os
import joblib
from scripts.explicabilidad import explicabilidad
from scripts.test import test

path = "DecisionTreeClassifier_0.9944"
model_path = os.path.join("modelos", path, f"{path}.pkl")
model_test = joblib.load(model_path)

test.main(model_test, path)
explicabilidad.main(model_test, path)