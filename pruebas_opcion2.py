import os
import joblib
from scripts.explicabilidad import explicabilidad
from scripts.test import test_opcion2

path = "DecisionTreeClassifier_0.9940"
model = os.path.join("modelos", path, f"DecisionTreeClassifier_0.9940.pkl")
model = joblib.load(model)

test_opcion2.main(model, path)
explicabilidad.main(model, path)