import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        z = X @ w + b        # shape (N,)
        p = _sigmoid(z)      # shape (N,)

        # Compute gradients
        error = p - y
        grad_w = (X.T @ error) / N
        grad_b = np.mean(error)

        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b