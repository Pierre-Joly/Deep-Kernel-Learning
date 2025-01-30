import numpy as np
from autograd import numpy as anp 

def xavier_uniform(rng, fan_in, fan_out, gain=1.0):
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_out, fan_in))

def init_params(X_train, y_train, input_dim=1, feature_dim=2, seed=123):
    rng = np.random.RandomState(seed)

    # Xavier initialization for weights (adjusted for tanh)
    gain_tanh = 5.0 / 3.0  # Compensates for tanh's variance reduction
    W1 = xavier_uniform(rng, input_dim, feature_dim, gain=gain_tanh)
    b1 = np.zeros(feature_dim)

    # --- Compute initial feature statistics ---
    phi_x = np.tanh(np.dot(X_train, W1.T) + b1)
    
    # Median pairwise distance of features
    pairwise_dist = np.sqrt(((phi_x[:, None, :] - phi_x[None, :, :])**2).sum(axis=-1))
    median_dist = np.median(pairwise_dist)
    
    # Variance of features
    feature_var = np.var(phi_x)

    # --- Scale weights to stabilize feature variance ---
    # Target variance = (median_dist^2) / 2 (prevents kernel collapse)
    target_var = (median_dist**2) / 2
    scale_factor = np.sqrt(target_var / (feature_var + 1e-6))  # Avoid division by zero
    
    W1_scaled = W1 * scale_factor
    b1_scaled = b1 * scale_factor

    # ---- Adaptive sigma initialization ----
    y_std = np.std(y_train)
    phi_std = np.std(phi_x)
    log_sigma = np.log(y_std / phi_std)  # Feature-normalized

    # --- Initialize GP hyperparameters ---
    log_lengthscale = np.log(median_dist)

    return {
        'W1': W1_scaled,
        'b1': b1_scaled,
        'log_lengthscale': log_lengthscale,
        'log_sigma': log_sigma,
    }

def forward(params, x):
    W1, b1 = params['W1'], params['b1']

    # Output
    phi_x = anp.tanh(anp.dot(x, W1.T) + b1)
    return phi_x

def rbf_kernel(params, X1, X2):
    phi_X1 = forward(params, X1)  # (N1, feature_dim)
    phi_X2 = forward(params, X2)  # (N2, feature_dim)
    
    lengthscale = anp.exp(params['log_lengthscale'])
    sigma       = anp.exp(params['log_sigma'])
    
    # Compute squared distance betwen each pair
    # ||phi_X1 - phi_X2||^2
    # ||a - b||^2 = (a^2 + b^2 - 2ab).
    
    # (N1,1,feature_dim), (1,N2,feature_dim) => diff shape: (N1, N2, feature_dim)
    diff = phi_X1[:, None, :] - phi_X2[None, :, :]
    sqdist = anp.sum(diff**2, axis=2)  # (N1, N2)
    
    K = sigma**2 * anp.exp(-0.5 * sqdist / (lengthscale**2))
    return K

def cov_matrix(params, X):
    K = rbf_kernel(params, X, X)
    N = X.shape[0]
    eps = 1e-6
    return K + eps * anp.eye(N)

def pack_params(params):
    W1 = params['W1'].ravel()
    b1 = params['b1'].ravel()
    
    return anp.concatenate([
        W1, b1,
        anp.array([params['log_lengthscale']]),
        anp.array([params['log_sigma']])
    ])

def unpack_params(theta, input_dim=1, feature_dim=2):
    # Sizes
    W1_size = feature_dim * input_dim
    b1_size = feature_dim
    
    idx1 = 0
    idx2 = W1_size
    W1 = theta[idx1:idx2].reshape(feature_dim, input_dim)
    
    idx1 = idx2
    idx2 = idx2 + b1_size
    b1 = theta[idx1:idx2]
    
    # Kernel hyperparameters : log_lengthscale, log_sigma, log_noise
    log_lengthscale = theta[idx2]
    log_sigma       = theta[idx2+1]
    
    return {
        'W1': W1,
        'b1': b1,
        'log_lengthscale': log_lengthscale,
        'log_sigma': log_sigma,
    }

def predict(params, X_train, y_train, X_star):
    K = cov_matrix(params, X_train)  # (N, N)
    Ks = rbf_kernel(params, X_star, X_train)  # (N_star, N)
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # mean = Ks * K^-1 * y
    mean_star = np.dot(Ks, alpha)
    
    # var = Kss - Ks K^-1 Ks^T
    # Kss = cov(X_star, X_star)
    Kss = cov_matrix(params, X_star)
    
    v = np.linalg.solve(L, Ks.T)
    var_star = np.diag(Kss) - np.sum(v**2, axis=0)  # (N_star, )
    
    return mean_star.ravel(), var_star.ravel()