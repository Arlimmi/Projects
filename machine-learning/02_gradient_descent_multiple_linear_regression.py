# Gradient descend on a multiple linear regression model

import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Model: f_{w,b}(x) = w * x + b
# With mulitple linear regression, w and x are vectors, and the model is defined as: 
# f_{w vector,b}(x vector) = w vector * x vector + b 

# Sample data
x = np.array([[1, 2],
              [2, 3],
              [3, 4]])
y = np.array([5, 7, 9])

m = 3
w = np.array([0.5, 0.5])  # initial weights, randomly chosen
b = 0  # initial bias, randomly chosen
learning_rate = 0.1 # Alpha

# Gradient Descend Algorithm
for i in range(3000):
    y_hat = np.dot(x, w) + b
    
    # Cost Function
    J = 1/(2*m) * np.sum((y_hat - y) ** 2)
    
    dj_dw = np.zeros(2)
    for j in range(2):
        dj_dw[j] = 1/m * np.sum((y_hat - y) * x[:, j])
    dj_db = 1/m * np.sum(y_hat - y)
    
    # Update weights and bias -- Gradient Descend Algorithm
    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db
    
print(f"Final: w = {w}, b = {b:.2f}")


