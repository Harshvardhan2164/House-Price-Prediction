# House Price Prediction

This repository contains a Python script that demonstrates the process of predicting house prices using various linear regression models. The script utilizes the `pandas`, `numpy`, `matplotlib`, `seaborn`, and `termcolor` packages for data processing, visualization, and text customization.

## Overview

The script follows the following steps:

1. **Importing Packages**: Importing the necessary packages for data processing, visualization, and modeling.

2. **Importing Data**: Reading the housing data from a CSV file and performing initial data preprocessing.

3. **Exploratory Data Analysis (EDA)**: Conducting exploratory data analysis by checking for missing values, summary statistics, and data types of each column. Converting relevant columns to numeric types.

4. **Data Visualization**: Creating visualizations to understand the relationships between features and the target variable using heatmap, scatter plots, and distribution plots.

5. **Feature Selection & Data Split**: Selecting the independent variables (features) and the dependent variable (target) for the prediction model. Splitting the data into training and testing sets.

6. **Modeling**: Building and training four different types of linear regression models, including Ordinary Least Squares (OLS), Ridge regression, Lasso regression and Bayesian regression. Making predictions on the testing data for each model.

7. **Evaluation**: Comparing the performance of each model by calculating the R-Squared Error, Explained Variance Score and visualizing the results using a bar plot.

8. **SalePrice Prediction Visualization**: Visualizing the predicted sale prices compared to the actual prices for both the training and testing datasets using scatter plots.

## Usage

1. Clone the repository:
git clone https://github.com/your-username/house-price-prediction.git

2. Navigate to the project directory:
cd house-price-prediction

3. Install the required packages:
pip install -r requirements.txt

4. Place your housing data CSV file in the project directory.

5. Open the `house_price_prediction.py` script and update the following line with your CSV file's name:
'''python
  df = pd.read_csv('house[1].csv')'''

6. Run the script:
python house_price_prediction.py

7. The script will generate various visualizations, including a heatmap, scatter plots, and a distribution plot, as well as print the summary statistics and evaluation metrics for each model.

8. The model comparison bar plot and the scatter plots of the predicted sale prices can be found in the project directory.

## Notes
1. Ensure that your data is properly formatted and does not contain missing values.
2. Additional preprocessing and feature engineering may be required based on your specific dataset.
3. The script currently uses a predefined set of features for the prediction models. Modify the X_var variable to include the desired independent variables.
4. The script uses a fixed random state for data splitting. Modify the random_state parameter in the train_test_split function for different random splits.
