from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('churn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0].tolist()

    return jsonify({
        'churn_prediction': int(prediction),
        'probability': round(proba[1]*100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
    