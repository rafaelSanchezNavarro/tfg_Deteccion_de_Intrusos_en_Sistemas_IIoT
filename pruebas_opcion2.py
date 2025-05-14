import os
import joblib
from scripts.explicabilidad import explicabilidad
from scripts.test import test_opcion2

path = "RandomForestClassifier_0.9960"
model = os.path.join("modelos", path, f"RandomForestClassifier_0.9960.pkl")
model = joblib.load(model)

# test_opcion2.main(model, path)
explicabilidad.main(model, path)