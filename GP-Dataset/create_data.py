import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
min = 0
max = 10

# Kernel
def square_exponential_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

#  Uniform Grid
X = np.linspace(min, max, N)
X = X[:, None]

# Covariance Matrix
length_scale = 1.0
sigma_f = 1.0
K = square_exponential_kernel(X, X, length_scale, sigma_f)

# Increase numerical stability
K += 1e-6 * np.eye(N)

# Mean function
mean = np.zeros(N)

# Simulate GP
GP = np.random.multivariate_normal(mean, K, size=3)

if __name__ == '__main__':
    plt.figure()
    for i in range(3):
        plt.plot(X, GP[i], label=f'Sample {i+1}')
    plt.title('Simulated Gaussian Process Samples')
    plt.xlabel('X')
    plt.ylabel('GP(m(x), k(x, x\'))')
    plt.legend()
    plt.show()
