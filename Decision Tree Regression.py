# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:05:15 2020

@author: Marcos_Coca_F.
"""
# Importing the libraries and the dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Assets/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression with whole dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# Prediting a new result

regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results ( Higher Resolution 2D )

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Thuth od Bluff ( Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


