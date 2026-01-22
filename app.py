from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Assets safely using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get 5 inputs from the form
        features = [
            float(request.form['radius']),
            float(request.form['texture']),
            float(request.form['perimeter']),
            float(request.form['area']),
            float(request.form['smoothness'])
        ]
        
        # Scale inputs
        final_features = np.array([features])
        scaled_features = scaler.transform(final_features)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        
        # Map 0/1 to labels (Sklearn breast cancer: 0=Malignant, 1=Benign)
        # Note: Often in medical datasets 0 is Benign and 1 is Malignant.
        # Sklearn default is: 'malignant' (0), 'benign' (1).
        result = "Benign (Safe)" if prediction == 1 else "Malignant (Danger)"
        css_class = "safe" if prediction == 1 else "danger"
        
        return render_template('index.html', prediction_text=f'{result}', css_class=css_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)