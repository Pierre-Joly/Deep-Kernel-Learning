import numpy as np
import matplotlib.pyplot as plt
from create_data import gaussian_process, square_exponential_kernel
from sklearn.model_selection import train_test_split
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

if __name__ == '__main__':
    ### Simulate Data
    np.random.seed(42)

    # Parameters
    N = 400
    x_min, x_max = 0, 10

    #  Uniform Grid
    X = np.linspace(x_min, x_max, N)

    # Kernel and mean functions
    length_scale = 1.0
    sigma_f = 1.0
    sigma_n = 0.1
    kernel_func = lambda x1, x2: square_exponential_kernel(x1, x2, length_scale, sigma_f, sigma_n)
    mean_func = lambda x: np.zeros(x.shape[0])

    # Simulate GP
    Y = gaussian_process(X, mean_func, kernel_func)

    ### Split Data
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.5)

    ### Fit GP and Predict

    # Optimize Hyperparameters
    length_scale, sigma_f, sigma_n = optimize_hyperparameters(X_train, Y_train)
    kernel_func = lambda x1, x2: square_exponential_kernel(x1, x2, length_scale, sigma_f, sigma_n)
    print(f"Optimized length_scale: {length_scale:.3f}, sigma_f: {sigma_f:.3f}, sigma_n: {sigma_n:.3f}")

    mean_post, cov_post = fit_gaussian_process(X_train, Y_train, X, kernel_func)

    ### Display Results

    # Sort X_test for Plotting
    idx = np.argsort(X)
    X = X[idx]
    mean_post = mean_post[idx]
    cov_post= cov_post[np.ix_(idx, idx)]

    # Plot
    plt.figure()
    plt.plot(X_train, Y_train, label=f'Training Data', marker='o', linestyle='None')
    plt.plot(X, mean_post, label=f'Posterior Mean')
    plt.fill_between(
        X.ravel(),
        mean_post - 1.96 * np.sqrt(np.diag(cov_post)),
        mean_post + 1.96 * np.sqrt(np.diag(cov_post)),
        alpha=0.2,
        color="blue",
        label="95% Confidence Interval",
    )
    plt.title('Simulated Gaussian Process Samples')
    plt.xlabel('X')
    plt.ylabel('GP(m(x), k(x, x\'))')
    plt.legend()
    plt.show()
