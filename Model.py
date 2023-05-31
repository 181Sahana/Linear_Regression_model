import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def Linear_Regression_Model(data, x, y):

    print(data.head())
    print(data.info())
    print(data.describe())

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x, y)

    # to calculate rot mean square
    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")

    # to calculate y intercept and slope
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    # to print the model summary
     #print(model.summary())
    

    y_pred = model.predict(x)
    print(f"predicted response:\n{y_pred}")

data = pd.read_csv('1_linear_regression_data.csv')
X = data[['youtube','facebook','newspaper']]
y = data['sales']
Linear_Regression_Model(data, X, y)