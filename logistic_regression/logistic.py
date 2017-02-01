'''Implementation of logistic regression with numpy'''

import numpy as np
import matplotlib.pyplot as plt

# We generate 20 points
x = np.linspace(-10, 10)
eps = np.random.normal(scale=3.0, size=50)

beta = np.array([1.0, -2.0])
linear = 1 * beta[0] + np.dot(beta[1], x) + eps
p_y = np.exp(linear) / (1 + np.exp(linear))
y = p_y > 0.5

plt.plot(x[y], y[y] - 0.5, '|', markersize=3.0)
plt.plot(x[~y], y[~y] + 0.5, '|', color='r', markersize=3.0)
plt.show()
