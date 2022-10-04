import joblib
from flask import Flask, request
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load(r'clf_model')

@app.route('/result', methods=['POST'])

def predict():
    data = request.json
    query_df = pd.DataFrame(data, index=[0])

    prediction = model.predict(query_df)
    prediction_flatten = np.hstack(prediction)

    print(prediction_flatten.tolist())
    return prediction_flatten.tolist()

if __name__ == '__main__' :
    app.run(debug=True)

