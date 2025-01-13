import numpy as np
import matplotlib.pyplot as plt

# Square Exponential Kernel
def square_exponential_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    x1 = x1[:, None] if x1.ndim == 1 else x1
    x2 = x2[:, None] if x2.ndim == 1 else x2
    sqdist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

def create_kernel(length_scale, sigma_f):
    return lambda x1, x2: square_exponential_kernel(x1, x2, length_scale, sigma_f)

def create_mean_func(mean):
    return lambda x: np.full(x.shape[0], mean)

# Gaussian Process Simulation
def gaussian_process(X, mean_func, kernel_func, sigma_n):
    mean = mean_func(X)
    K = kernel_func(X, X) + sigma_n**2 * np.eye(len(X))
    return np.random.multivariate_normal(mean, K)

if __name__ == '__main__':
    # Parameters
    N = 100
    x_min, x_max = 0, 10
    number_of_trajectories = 3

    #  Uniform Grid
    X = np.linspace(x_min, x_max, N)

    # Covariance Matrix
    length_scale = 1.0
    sigma_f = 1.0
    sigma_n = 0.0
    kernel_func = create_kernel(length_scale, sigma_f)
    mean_func = create_mean_func(0.0)

    # Simulate GP
    GP = gaussian_process(X, mean_func, kernel_func, sigma_n)

    plt.figure()
    plt.plot(X, GP, label=f'Trajectory')
    plt.title('Simulated Gaussian Process Samples')
    plt.xlabel('X')
    plt.ylabel('GP(m(x), k(x, x\'))')
    plt.legend()
    plt.show()
