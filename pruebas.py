import os
import joblib
from scripts.test import test

path = "RandomForestClassifier_0.9908"
model_path = os.path.join("modelos", path, f"{path}.pkl")
model_test = joblib.load(model_path)
print(model_test)
test.main(model_test, path)