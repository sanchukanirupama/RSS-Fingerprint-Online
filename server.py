from unittest import result
import joblib
from flask import Flask, request
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load(r'clf_model')

@app.route('/', methods=['GET'])
def respond():
    return "test route - health good!"

@app.route('/predict-position', methods=['POST'])

def predict():
    try:
        data = request.json
        query_df = pd.DataFrame(data, index=[0])

        prediction = model.predict(query_df)
        prediction_flatten = np.hstack(prediction)

        x = prediction_flatten.tolist()[0]
        y = prediction_flatten.tolist()[1]

        data['x'] = x
        data['y'] = y

        print (data)
        return data

    except Exception as e:
        print(e)
        print('JSON Structure is wrong')
        return request.json
        

    

if __name__ == '__main__' :
    app.run(debug=True)

