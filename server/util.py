import joblib 
import numpy as np
import pandas as pd

__columns = None
__model = None

def get_predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, thal):
    if not all(isinstance(i, (int, float)) for i in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, thal]):
        raise ValueError("All inputs must be numeric.")

    if __model is None:
        raise RuntimeError("Model is not loaded. Please call load_saved() before making predictions.")

    new_sample = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, thal]]
    x = pd.DataFrame(new_sample, columns=__columns)

    predict = str(__model.predict(x)[0])
    result = "No" if predict == "0" else "Yes"
    return result

def load_saved():
    print('Loading saved model and columns...')

    global __columns
    __columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "thal"]

    global __model
    try:
        with open('/Users/macbook/Documents/Project/MachineLearning/Heart-Prediction/knn_model.sav', 'rb') as f:
            __model = joblib.load(f)
        print("Model and columns loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please check the file path.")
        __model = None

if __name__ == '__main__':
    load_saved()

