# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:51:26 2020

@author: Marcos_Coca_F.
"""

# Support Vector Regression (SVR)

# Importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Impoting dataset

dataset = pd.read_csv('/Users/Alien/Desktop/Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 2/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Training the model

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

# Predicting a new result

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# Visualising the SVR

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Truth or Bluff ( Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Smother line

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or Bluff ( Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
        