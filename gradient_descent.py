# Gradient descend on a univariate linear regression model

import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Model: f_{w,b}(x) = w * x + b

# Sample data
x = np.array([1, 2, 3])
y = np.array([5, 7, 9])

m = 3
w = 0.5  # initial weight, randomly chosen
b = 0  # initial bias, randomly chosen
learning_rate = 0.1 # Alpha

# Gradient Descend Algorithm
for i in range(3000):
    y_hat = w * x + b
    # Cost Function
    J = 1/(2*m) * np.sum((y_hat - y) ** 2)
   
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {J}, w = {w}, b = {b}")
    
    dj_dw = 1/m * np.sum((y_hat - y) * x)
    dj_db = 1/m * np.sum(y_hat - y)
    
    # Update weights and bias -- Gradient Descend Algorithm
    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db
    
print(f"Final: w = {w:.2f}, b = {b:.2f}")