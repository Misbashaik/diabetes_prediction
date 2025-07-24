from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    return render_template('index.html', prediction_text=f'Result: {output}')

if __name__ == "__main__":
    app.run(debug=True)
