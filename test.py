# Gradient descend on a univariate linear regression model

import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Model: f_{w,b}(x) = w * x + b

# Sample data
x = np.array([1, 2, 3])
y = np.array([3, 5, 7])

m = 3
w = 0.5  # initial weight, randomly chosen
b = 0  # initial bias, randomly chosen
learning_rate = 0.1 # Alpha

cost_history = []
w_history = []
b_history = []

for i in range(3000):
    y_hat = w * x + b
    J = 1/(2*m) * np.sum((y_hat - y) ** 2)

    cost_history.append(J)   # must be indented inside the loop
    w_history.append(w)
    b_history.append(b)

    if i % 500 == 0:
        print(f"Iteration {i}: Cost = {J:.3e}, w = {w:.6f}, b = {b:.6f}")

    dj_dw = 1/m * np.sum((y_hat - y) * x)
    dj_db = 1/m * np.sum(y_hat - y)

    w = w - learning_rate * dj_dw
    b = b - learning_rate * dj_db

print(len(w_history))   # should print 3000

print(f"Final: w = {w:.2f}, b = {b:.2f}")

# --- Cost surface grid ---
w_vals = np.linspace(-1, 4, 200)
b_vals = np.linspace(-2, 4, 200)
W, B = np.meshgrid(w_vals, b_vals)

J_grid = np.zeros_like(W)
for xi, yi in zip(x, y):
    J_grid += (W * xi + B - yi) ** 2
J_grid /= (2 * m)

fig = plt.figure(figsize=(16, 4.5))

# 1. Data and fit
ax0 = fig.add_subplot(1, 4, 1)
ax0.scatter(x, y, color="red", label="Training data")
x_line = np.linspace(x.min(), x.max(), 100)
ax0.plot(x_line, w * x_line + b, label=f"y = {w:.2f}x + {b:.2f}")
ax0.set_title("Fit")
ax0.legend()

# 2. Cost vs iterations
ax1 = fig.add_subplot(1, 4, 2)
ax1.plot(cost_history)
ax1.set_yscale("log")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cost J")
ax1.set_title("Convergence")

# 3. Contours with descent path
ax2 = fig.add_subplot(1, 4, 3)
ax2.contour(W, B, J_grid, levels=np.logspace(-2, 1.5, 20), cmap="viridis")
ax2.plot(w_history[:200], b_history[:200], "r.-", markersize=3, label="Path")
ax2.plot(w_history[0], b_history[0], "go", label="Start")
ax2.plot(w, b, "r*", markersize=12, label="End")
ax2.set_xlabel("w")
ax2.set_ylabel("b")
ax2.set_title("Cost contours")
ax2.legend()

# 4. 3D bowl
ax3 = fig.add_subplot(1, 4, 4, projection="3d")
ax3.plot_surface(W, B, J_grid, cmap="viridis", alpha=0.8)
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_zlabel("J")
ax3.set_title("Cost surface")

plt.tight_layout()
plt.show()