# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:25:23 2020

@author: Marcos_Coca_F.
"""

# Polynomial Regression

# Importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impoting dataset

dataset = pd.read_csv('/Users/Alien/Desktop/Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the linear regression model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the polynomial regression model

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the linear regressions

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff ( Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial regressions

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Thrth or Bluff ( Polynomial Regrassion)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Smother curve

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff ( Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression

lin_reg.predict([[6.5]])

# Predicting a new result with polynomial regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))












































