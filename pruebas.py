import os
import joblib
from scripts.test import test

path = "RandomForestClassifier_0.9962"
model_path = os.path.join("modelos", path, f"{path}.pkl")
model_test = joblib.load(model_path)
test.main(model_test, path)