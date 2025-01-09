import numpy as np
import matplotlib.pyplot as plt

# Square Exponential Kernel
def square_exponential_kernel(X, length_scale=1.0, sigma_f=1.0, noise=1e-6):
    sqdist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    K = sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)
    K += noise * np.eye(X.shape[0])
    return K

# Gaussian Process
def gaussian_process(X, mean, kernel, size=1):
    M = mean(X)
    K = kernel(X)
    return np.random.multivariate_normal(M, K, size=size)

if __name__ == '__main__':
    # Parameters
    N = 100
    x_min, x_max = 0, 10
    number_of_trajectories = 3

    #  Uniform Grid
    X = np.linspace(x_min, x_max, N)[:, None]

    # Covariance Matrix
    length_scale = 1.0
    sigma_f = 1.0
    kernel_func = lambda X: square_exponential_kernel(X, length_scale, sigma_f, noise=1e-6)
    mean_func = lambda x: np.zeros(x.shape[0])

    # Simulate GP
    GP = gaussian_process(X, mean_func, kernel_func, size=number_of_trajectories)

    plt.figure()
    for i in range(number_of_trajectories):
        plt.plot(X, GP[i], label=f'Sample {i+1}')
    plt.title('Simulated Gaussian Process Samples')
    plt.xlabel('X')
    plt.ylabel('GP(m(x), k(x, x\'))')
    plt.legend()
    plt.show()
