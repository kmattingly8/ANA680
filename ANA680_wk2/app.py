from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure the pickle file is in your project 
folder)
model = joblib.load("breast_cancer_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    var_1 = float(request.form['var_1'])
    var_2 = float(request.form['var_2'])
    var_3 = float(request.form['var_3'])
    var_4 = float(request.form['var_4'])
    var_5 = float(request.form['var_5'])

    # Create an array of input data
    input_data = np.array([[var_1, var_2, var_3, var_4, var_5]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Map prediction result to human-readable text
    result = "Malignant" if prediction == 1 else "Benign"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

