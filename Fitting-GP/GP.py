import numpy as np
import matplotlib.pyplot as plt
from create_data import gaussian_process, square_exponential_kernel, create_kernel, create_mean_func
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

JITTER = 1e-6

# Log Marginal Likelihood
def log_marginal_likelihood(params, X_train, Y_train):
    length_scale, sigma_f, sigma_n = params
    n = len(X_train)
    
    # Kernel matrix
    K = square_exponential_kernel(X_train, X_train, length_scale, sigma_f)
    K += (sigma_n**2 + JITTER) * np.eye(n)  # Add noise

    # Cholesky decomposition for stability
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros_like(params)  # Penalize invalid hyperparameters

    # Solve for alpha
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
    
    # Negative log-marginal likelihood
    log_likelihood = -0.5 * Y_train.T @ alpha
    log_likelihood -= np.sum(np.log(np.diag(L)))
    log_likelihood -= n / 2 * np.log(2 * np.pi)
    neg_log_likelihood = -log_likelihood  # For minimization

    return neg_log_likelihood

# Hyperparameter Optimization
def optimize_hyperparameters(X_train, Y_train):
    # Initial guess and bounds
    initial_params = np.array([np.std(X_train), np.std(Y_train), 0.1])
    bounds = [(1e-3, None), (1e-3, None), (1e-6, None)]
    
    # Optimization with gradient
    result = minimize(
        lambda params: log_marginal_likelihood(params, X_train, Y_train),
        initial_params,
        bounds=bounds,
        method="L-BFGS-B"
    )

    if not result.success:
        print("Warning: Optimization failed. Using initial parameters.")
        return initial_params
    
    return result.x

# Fit Gaussian Process
def fit_gaussian_process(X_train, Y_train, X_test, kernel_func, sigma_n):
    K = kernel_func(X_train, X_train) + (sigma_n**2 + JITTER) * np.eye(len(X_train))
    K_s = kernel_func(X_train, X_test)
    K_ss = kernel_func(X_test, X_test) + sigma_n**2 * np.eye(len(X_test))

    # Cholesky decomposition: K = L @ L.T
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not positive definite. Check kernel or noise parameters.")

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))

    # Compute posterior mean
    mean_post = K_s.T @ alpha

    # Solve for v: v = L^-1 * K_s
    v = np.linalg.solve(L, K_s)

    # Compute posterior covariance
    cov_post = K_ss - v.T @ v

    # Ensure symmetry of covariance matrix
    cov_post = (cov_post + cov_post.T) / 2

    return mean_post, cov_post

def simulate_data(N, x_min, x_max, length_scale, sigma_f, sigma_n):
    X = np.linspace(x_min, x_max, N)
    kernel_func = create_kernel(length_scale, sigma_f)
    mean_func = create_mean_func(0.0)
    Y = gaussian_process(X, mean_func, kernel_func, sigma_n)
    return X, Y

def visualize_gp(X, X_train, Y_train, mean_post, cov_post):
    idx = np.argsort(X)
    X = X[idx]
    mean_post = mean_post[idx]
    cov_post = cov_post[np.ix_(idx, idx)]

    plt.figure()
    plt.plot(X_train, Y_train, 'o', label='Training Data', linestyle='None')
    plt.plot(X, mean_post, label='Posterior Mean')
    plt.fill_between(
        X.ravel(),
        mean_post - 1.96 * np.sqrt(np.diag(cov_post)),
        mean_post + 1.96 * np.sqrt(np.diag(cov_post)),
        alpha=0.2, color="blue", label="95% Confidence Interval"
    )
    plt.title('Gaussian Process Regression')
    plt.xlabel('Input (X)')
    plt.ylabel('GP Mean and Confidence')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Parameters
    np.random.seed(42)
    N = 400
    x_min, x_max = 0, 10
    length_scale, sigma_f, sigma_n = 1.0, 1.0, 0.1

    # Simulate Data
    X, Y = simulate_data(N, x_min, x_max, length_scale, sigma_f, sigma_n)

    # Split Data
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.5, random_state=42, shuffle=True)

    # Optimize Hyperparameters
    length_scale, sigma_f, sigma_n = optimize_hyperparameters(X_train, Y_train)
    kernel_func = create_kernel(length_scale, sigma_f)
    print(f"Optimized length_scale: {length_scale:.3f}, sigma_f: {sigma_f:.3f}, sigma_n: {sigma_n:.3f}")

    # Fit GP
    mean_post, cov_post = fit_gaussian_process(X_train, Y_train, X, kernel_func, sigma_n)

    # Visualize Results
    visualize_gp(X, X_train, Y_train, mean_post, cov_post)