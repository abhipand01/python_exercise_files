import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pylab import rcParams

################################ Linear Regression ##############################################################

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

rcParams['figure.figsize'] = 10, 8

# Artificially creating features and data (predictors and predicted)

np.random.seed(25)
rooms = 2*np.random.rand(100, 1) + 3
rooms[1:10]

price = 265 + 6*rooms + abs(np.random.randn(100, 1))
price[1:10]

plt.plot(rooms, price, 'r^')
plt.xlabel("# of Rooms, 2019 Average")
plt.ylabel("2019 Average Home. 1000s USD")
plt.show()


#### Setting up the regression model
linreg = LinearRegression().fit(rooms, price)
print(linreg.intercept_, linreg.coef_)

# model performance
linreg.score(rooms, price)
