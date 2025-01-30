import numpy as np


def generate_data(n=30, noise_std=0.1, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n)
    y = np.sin(X) + noise_std * np.random.randn(n)
    return X.reshape(-1, 1), y.reshape(-1, 1)
