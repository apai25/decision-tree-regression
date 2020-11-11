import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# importing the datasets
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# no data preprocessing necessary

# creating and training the regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='mse', random_state=0)
regressor.fit(x, y)

# predicting with the regressor
predictions = regressor.predict(x)

# more precise graphing
x_grid = np.arange(min(x), max(x) + .01, 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

# visualizing the model
plt.scatter(x, y, color='red', label='data points')
plt.plot(x_grid, regressor.predict(x_grid), color='blue', label='model representation')
plt.legend()
plt.show()