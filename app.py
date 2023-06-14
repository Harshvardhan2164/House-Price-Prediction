from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read the housing data from CSV file
    df = pd.read_csv('house.csv')
    df.set_index('Id', inplace=True)
    
    # Data preprocessing
    df.dropna(inplace=True)
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'], errors='coerce')
    df['MasVnrArea'] = df['MasVnrArea'].astype('int64')

    # Independent variable declaration
    X_var = df[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']].values
    # Dependent(Target) variable declaration
    y_var = df['SalePrice'].values

    # Splitting the data into train(80) and test(20) sets
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=0)

    # Create an instance of the LinearRegression model
    model = LinearRegression()
    # Train the model on training data
    model.fit(X_train, y_train)

    # Extract the input values from the form
    features = [float(x) for x in request.form.values()]

    # Perform prediction using the model
    prediction = model.predict([features])[0]

    # Round the prediction to two decimal places
    prediction = round(prediction, 2)

    # Render the result template and pass the prediction value
    return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
