import os
import joblib
from scripts.test import test

path = "DecisionTreeClassifier_0.9911"
model_path = os.path.join("modelos", path, f"{path}.pkl")
model_test = joblib.load(model_path)
test.main(model_test, path)