import numpy as np

# Direct Formula Method
def direct_formula(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    a = (sum_y - b * sum_x) / n

    return a, b


# Matrix Method (Normal Equation)
def matrix_method(x, y):
    # Add bias term (column of 1s)
    X_b = np.c_[np.ones(len(x)), x]  # shape: (n, 2)

    # Normal Equation
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    a = beta[0]  # intercept
    b = beta[1]  # slope

    return a, b


# Gradient Descent Method
def gradient_descent(x, y, lr=0.01, epochs=1000):
    n = len(x)
    
    a = 0
    b = 0

    losses = []

    for _ in range(epochs):
        y_pred = a + b * x

        # Loss (MSE)
        loss = (1/n) * np.sum((y - y_pred)**2)
        losses.append(loss)

        # Gradients
        da = (-2/n) * np.sum(y - y_pred)
        db = (-2/n) * np.sum(x * (y - y_pred))

        # Update
        a -= lr * da
        b -= lr * db

    return a, b, losses

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)