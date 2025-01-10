import numpy as np
from create_data import square_exponential_kernel
from scipy.optimize import minimize

# Log-Marginal Likelihood
def log_marginal_likelihood(params, X_train, y_train):
    length_scale, sigma_f, sigma_n = params
    K = square_exponential_kernel(X_train, X_train, length_scale, sigma_f, sigma_n)
    N = len(X_train)
    
    # Compute log marginal likelihood
    K_inv = np.linalg.inv(K)
    log_likelihood = -0.5 * y_train.T @ K_inv @ y_train
    log_likelihood -= 0.5 * np.log(np.linalg.det(K))
    log_likelihood -= N / 2 * np.log(2 * np.pi)
    
    return -log_likelihood

# Optimize Hyperparameters
def optimize_hyperparameters(X_train, y_train):
    # Initial guesses for length_scale, sigma_f, sigma_n
    initial_params = [1.0, 1.0, 0.1]
    bounds = [(1e-3, None), (1e-3, None), (1e-6, None)]  # Positive constraints
    
    result = minimize(
        log_marginal_likelihood, 
        initial_params, 
        args=(X_train, y_train), 
        bounds=bounds, 
        method="L-BFGS-B"
    )
    
    return result.x

# Fit Gaussian Process
def fit_gaussian_process(X_train, Y_train, X_test, kernel_func):
    # Compute kernels
    K = kernel_func(X_train, X_train) # Train covariance
    K_s = kernel_func(X_train, X_test)  # Cross-covariance
    K_ss = kernel_func(X_test, X_test)  # Test covariance

    # Compute posterior mean
    K_inv = np.linalg.inv(K)
    mean_post = K_s.T @ K_inv @ Y_train

    # Compute posterior covariance
    cov_post = K_ss - K_s.T @ K_inv @ K_s

    return mean_post, cov_post