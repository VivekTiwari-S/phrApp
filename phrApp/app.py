from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
with open('health_api.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Category mapping: convert dropdown values to numerical codes
category_mapping = {
    "beverages": 0,
    "dairy": 1,
    "snacks": 2
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    category = request.form['category']
    sugar = float(request.form['sugar'])
    fat = float(request.form['fat'])
    calories = float(request.form['calories'])
    sodium = float(request.form['sodium'])
    protein = float(request.form['protein'])
    fiber = float(request.form['fiber'])

    # Convert category to numerical code
    category_code = category_mapping.get(category)

    # Prepare data for prediction with all 7 features
    data = np.array([[category_code, sugar, fat, calories, sodium, protein, fiber]])

    # Scale the input data
    data_scaled = scaler.transform(data)

    # Predict the health rating
    rating = round(model.predict(data_scaled)[0], 1)

    return render_template('result.html', rating=rating)

if __name__ == '__main__':
    app.run(debug=True)