# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models from the pickle file
with open('models.pkl', 'rb') as file:
    models = pickle.load(file)


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form.to_dict()

    # Print the form data for debugging
    print("Form Data:", data)

    # Check if 'region' is in the data
    if 'region' not in data:
        return render_template('index.html', prediction_text='Error: Region data is missing.')

    # Convert the form data into a DataFrame (assuming all input fields are strings)
    data_df = pd.DataFrame([data])
    
    # Drop the 'model' field since it's not part of the input features
    if 'model' in data_df.columns:
        data_df = data_df.drop('model', axis=1)
    
    # Perform preprocessing
    if 'sex' in data_df.columns:
        data_df['sex'] = data_df['sex'].replace({'female': 0, 'male': 1})
    
    if 'smoker' in data_df.columns:
        data_df['smoker'] = data_df['smoker'].replace({'no': 0, 'yes': 1})

    # One-hot encoding for region
    if 'region' in data_df.columns:
        region_dummies = pd.get_dummies(data_df['region'], prefix='region').astype(int)
        data_df = pd.concat([data_df, region_dummies], axis=1)
        data_df = data_df.drop('region', axis=1)
    else:
        # Handle missing 'region' in data_df
        region_dummies = pd.DataFrame(columns=['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'])
        data_df = pd.concat([data_df, region_dummies], axis=1)
    
    # Fill missing columns with 0 (this assumes that any missing region dummy variables are simply not applicable)
    for col in ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']:
        if col not in data_df.columns:
            data_df[col] = 0
            
    # Standardize the data
    data_df_scaled = scaler.transform(data_df)
    selected_model = request.form.get('model')
    

    # Select the model based on user input (e.g., via a dropdown menu in the form)
    selected_model = request.form.get('model')
    
    if selected_model not in models:
        return render_template('index.html', prediction_text='Error: Selected model is not available.')

    # Make prediction
    prediction = models[selected_model].predict(data_df_scaled)
    
    # Return the result to the user
    return render_template('index.html', prediction_text=f'Estimated Health Price: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
