import numpy as np
import matplotlib.pyplot as plt
from src.regression import *

# Dataset
np.random.seed(0)
X = np.linspace(0, 10, 50)
y = 3 * X + 5 + np.random.randn(50)


a_d, b_d = direct_formula(X, y)

print("Intercept:", a_d)
print("Slope:", b_d)




a_m, b_m = matrix_method(X, y)

print("\n[Matrix Method]")
print("Intercept:", a_m)
print("Slope:", b_m)



# High learning rate
a_fast, b_fast, losses_fast = gradient_descent(X, y, lr=0.005)

# Low learning rate
a_slow, b_slow, losses_slow = gradient_descent(X, y, lr=0.001)

# Prints
print("\n--- Learning Rate Comparison ---")
print("High LR (0.1):", a_fast, b_fast)
print("Low LR (0.001):", a_slow, b_slow)

print("\n--- FINAL COMPARISON ---")
print("Direct Method:", a_d, b_d)
print("Matrix Method:", a_m, b_m)
print("Gradient Descent:", a_fast, b_fast)

print("\n[Gradient Descent]")
print("Intercept:", a_fast)
print("Slope:", b_fast)

print("\n--- ERROR COMPARISON ---")
print("Direct MSE:", mse(y, a_d + b_d*X))
print("Matrix MSE:", mse(y, a_m + b_m*X))
print("Gradient MSE:", mse(y, a_fast + b_fast*X))



#loss plot
plt.plot(losses_fast, label='LR=0.1')
plt.plot(losses_slow, label='LR=0.001')
plt.legend()
plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("results/loss.png")
plt.close()

# Plot data + fitted line
sorted_idx = np.argsort(X)
X_sorted = X[sorted_idx]
y_sorted_pred = a_fast + b_fast * X_sorted

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_sorted, y_sorted_pred, color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Regression Fit")
plt.savefig("results/fit.png")
plt.close()


