# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization for plotting graphs
import seaborn as sb # visualization of statistal data
from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS(Ordinary Least Squares) algorithm

from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Least Absolute Shrinkage and Selection Operator algorithm
''' Lasso and Ridge models are used for regularization in order to 
    overcome overfitting and reduce complexity.
    1. (L1 regularization)Lasso uses the magnitude of the coefficients and introduces a penalization factor.
    2. (L2 regularization)Ridge also performs in similar way but uses the square of the coefficients.
'''

from sklearn.linear_model import BayesianRidge # Bayesian algorithm
''' Bayesian regression is better than OLS as it uses prior knowledge about the data to learn more and create
    more accurate predictions. It is an choice when data is complex.
    It has a probabilistic character, it can produce more accurate estimates for regression parameters.
'''

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric
# from sklearn.externals import joblib

sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size
'''These lines set the style and size of the plots created using Seaborn and Matplotlib'''

# IMPORTING DATA

df = pd.read_csv('house[1].csv') # Read the housing data from CSV file
df.set_index('Id', inplace = True) # Sets the 'ID' column as the index

print(df.head(5)) # Prints first 5 rows of the dataset

# EDA (Exploratory Data Analysis)

df.dropna(inplace = True) # Drops any rows with missing values from the Dataframe

print(cl(df.isnull().sum(), attrs = ['bold'])) # Print the number of missing values for each column

print(df.describe()) # Print summary statistics(mean, count, standard deviation, min, max, quartiles) of the dataset

print(cl(df.dtypes, attrs = ['bold'])) # Print the data types of each column

df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'], errors = 'coerce') # Convert 'MasVnrArea' to numeric type
df['MasVnrArea'] = df['MasVnrArea'].astype('int64') # Converts to 'int64' type

print(cl(df.dtypes, attrs = ['bold'])) # Print the data types of each column

# DATA VISUALIZATION

# 1. Heatmap

sb.heatmap(df.corr(), annot = True, cmap = 'magma') # Create a heatmap to visualize the correlation between features

plt.savefig('heatmap.png') # Saves the heatmap plot as an image
plt.show() # Displays the heatmap plot

# 2. Scatter plot
''' Scatter plot is also useed to observe linear relations between two variables in dataset.
    The below lines of code defines a function 'scatter_df' that creates scatter plots of each 
    feature against target variable 'SalePrice' and saves them as images. The function is then 
    called with 'SalePrice' as the argument to create the scatter plots.
'''

def scatter_df(y_var):
    # scatter_df = df.drop(y_var, axis = 1)
    i = df.columns
    
    plot1 = sb.scatterplot(x = i[0], 
                           y = y_var, 
                           data = df, 
                           color = 'orange', 
                           edgecolor = 'b', 
                           s = 150) # Define the scatterplot between the target variable y_var(SalePrice) and feature
    plt.title('{} / Sale Price'.format(i[0]), fontsize = 16) # Set the plot title
    plt.xlabel('{}'.format(i[0]), fontsize = 14) # Set the x-axis label
    plt.ylabel('Sale Price', fontsize = 14) # Set the y-axis label
    plt.xticks(fontsize = 12) # Set the x-axis tick labels font size
    plt.yticks(fontsize = 12) # Set the y-axis tick labels font size
    plt.savefig('scatter1.png')
    plt.show()
    
    plot2 = sb.scatterplot(x = i[1], 
                           y = y_var,
                           data = df,
                           color = 'yellow', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[1]), fontsize = 16)
    plt.xlabel('{}'.format(i[1]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter2.png')
    plt.show()
    
    plot3 = sb.scatterplot(x = i[2], 
                           y = y_var, 
                           data = df, 
                           color = 'aquamarine', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[2]), fontsize = 16)
    plt.xlabel('{}'.format(i[2]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter3.png')
    plt.show()
    
    plot4 = sb.scatterplot(x = i[3], 
                           y = y_var, 
                           data = df, 
                           color = 'deepskyblue', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[3]), fontsize = 16)
    plt.xlabel('{}'.format(i[3]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter4.png')
    plt.show()
    
    plot5 = sb.scatterplot(x = i[4], 
                           y = y_var, 
                           data = df, 
                           color = 'crimson', 
                           edgecolor = 'white', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[4]), fontsize = 16)
    plt.xlabel('{}'.format(i[4]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter5.png')
    plt.show()
    
    plot6 = sb.scatterplot(x = i[5], 
                           y = y_var, 
                           data = df, 
                           color = 'darkviolet', 
                           edgecolor = 'white', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[5]), fontsize = 16)
    plt.xlabel('{}'.format(i[5]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter6.png')
    plt.show()
    
    plot7 = sb.scatterplot(x = i[6], 
                           y = y_var, 
                           data = df, 
                           color = 'khaki', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[6]), fontsize = 16)
    plt.xlabel('{}'.format(i[6]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter7.png')
    plt.show()
    
    plot8 = sb.scatterplot(x = i[7], 
                           y = y_var, 
                           data = df, 
                           color = 'gold', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[7]), fontsize = 16)
    plt.xlabel('{}'.format(i[7]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter8.png')
    plt.show()
    
    plot9 = sb.scatterplot(x = i[8], 
                           y = y_var, 
                           data = df, 
                           color = 'r', 
                           edgecolor = 'b', 
                           s = 150)
    plt.title('{} / Sale Price'.format(i[8]), fontsize = 16)
    plt.xlabel('{}'.format(i[8]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter9.png')
    plt.show()
    
    plot10 = sb.scatterplot(x = i[9], 
                            y = y_var, 
                            data = df, 
                            color = 'deeppink', 
                            edgecolor = 'b', 
                            s = 150)
    plt.title('{} / Sale Price'.format(i[9]), fontsize = 16)
    plt.xlabel('{}'.format(i[9]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter10.png')
    plt.show()
    
scatter_df('SalePrice') # Dependent variable input 'SalePrice'

# 3. Distribution plot
# Shows the distribution of the 'SalePrice' variable in the dataset.
sb.distplot(df['SalePrice'], color = 'r') # Create a distribution plot of the 'SalePrice' variable
plt.title('Sale Price Distribution', fontsize = 16)
plt.xlabel('Sale Price', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('distplot.png')
plt.show()

# FEATURE SELECTION & DATA SPLIT

# Independent variable declaration
X_var = df[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']].values
# Dependent(Target) variable declaration
y_var = df['SalePrice'].values

# Splitting the data into train(80) and test(20) sets
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

# Printing the Train and Test samples of target variable and independent variable
print(cl('X_train samples : ', attrs = ['bold']), X_train[0:5])
print(cl('X_test samples : ', attrs = ['bold']), X_test[0:5])
print(cl('y_train samples : ', attrs = ['bold']), y_train[0:5])
print(cl('y_test samples : ', attrs = ['bold']), y_test[0:5])

# MODELING
''' We build and train 5 different types of linear regression models which are OLS model,
    Ridge regression model, Lasso regression model, and Bayesian regression model.
    The process for all the models are the same:
        1. We define a variable to store the model algorithm.
        2. We fit the train seet variables into the model.
        3. Make some predictions in the test set.
'''

# 1. OLS

ols = LinearRegression() # Create an instance of the LinearRegression model
ols.fit(X_train, y_train) # Train the model on training data
ols_pred = ols.predict(X_test) # Make predictions on the testing data

# 2. Ridge

ridge = Ridge(alpha = 0.5) # Create an instance of the Ridge model
ridge.fit(X_train, y_train) # Train the model on training data
ridge_pred = ridge.predict(X_test) # Make predictions on the testing data

# 3. Lasso

lasso = Lasso(alpha = 0.01) # Create an instance of the Lasso model
lasso.fit(X_train, y_train) # Train the model on training data
lasso_pred = lasso.predict(X_test) # Make predictions on the testing data

# 4. Bayesian

bayesian = BayesianRidge() # Create an instance of the BayesianRidge model
bayesian.fit(X_train, y_train) # Train the model on training data
bayesian_pred = bayesian.predict(X_test) # Make predictions on the testing data

# Save models using joblib
# joblib.dump(ols, 'ols_model.joblib')
# joblib.dump(ridge, 'ridge_model.joblib')
# joblib.dump(lasso, 'lasso_model.joblib')
# joblib.dump(bayesian, 'bayesian_model.joblib')


# PLOTTING THE SALEPRICE PREDICTION FROM TRAIN AND TEST DATASET

# 1. OLS
# Plotting the sale prices for the test data

training_data_prediction = ols.predict(X_test)
plt.scatter(y_test, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# 2. Ridge
# The sale prices for the test data

training_data_prediction = ridge.predict(X_test)
plt.scatter(y_test, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# 3. Lasso
# Plotting the sale prices for the test data

training_data_prediction = lasso.predict(X_test)
plt.scatter(y_test, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

# 4. Bayesian
# Plotting the sale prices for the test data

training_data_prediction = bayesian.predict(X_test)
plt.scatter(y_test, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()


# EVALUATION
# Comparing the 4 models to find the best model for the house price prediction

# 1. Explained Variance Score

# Explained variance score should be between 0.60 to 1
print(cl('EXPLAINED VARIANCE SCORE:', attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('Explained Variance Score of Ridge model is {}'.format(evs(y_test, ridge_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('Explained Variance Score of Lasso model is {}'.format(evs(y_test, lasso_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('Explained Variance Score of Bayesian model is {}'.format(evs(y_test, bayesian_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

# 2. R-squared

print(cl('R-SQUARED:', attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')

print(cl('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_pred)), attrs = ['bold']))

print('-------------------------------------------------------------------------------')
