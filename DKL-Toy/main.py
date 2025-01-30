import numpy as np
from create_data import generate_data
from model import init_params, predict
from optimization import (
    train_gradient_descent_with_trace,
    train_lbfgsb_with_trace,
    pack_params
)
from plotting import (
    plot_results,
    plot_trajectories,
    compute_nll_for_param_set,
    sample_param_space)

def main():
    ### Data
    X_train, y_train = generate_data(n=30, noise_std=0)

    ### Hyperparameters
    input_dim = 1
    feature_dim = 2
    base_params = init_params(X_train, y_train, input_dim, feature_dim)
    
    param_samples = sample_param_space(n_samples=1000, input_dim=input_dim, feature_dim=feature_dim)
    nll_values = compute_nll_for_param_set(X_train, y_train, param_samples, input_dim=input_dim, feature_dim=feature_dim)
    
    ### Optimization
    sgd_params_init = {}
    for k, v in base_params.items():
        sgd_params_init[k] = np.copy(v) if isinstance(v, np.ndarray) else v

    # SGD
    final_sgd_params, sgd_params_traj = train_gradient_descent_with_trace(X_train, y_train, sgd_params_init, lr=1e-1, n_epochs=15)

    # L-BFGS
    lbfgs_params_init = pack_params(base_params)
    final_lbfgs_params, lbfgs_params_traj = train_lbfgsb_with_trace(X_train, y_train, lbfgs_params_init, input_dim, feature_dim)
    
    ### Visualization
    # Gradient landscape
    #plot_trajectories(param_samples, sgd_params_traj, lbfgs_params_traj, nll_values, reduction="umap")
    plot_trajectories(param_samples, sgd_params_traj, lbfgs_params_traj, nll_values, reduction="pca")

    # Prediction 
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    mu_lbfgs, var_lbfgs = predict(final_lbfgs_params, X_train, y_train, X_test)
    mu_sgd, var_sgd = predict(final_sgd_params, X_train, y_train, X_test)
    
    plot_results(X_train, y_train, X_test,
                 mu_lbfgs, var_lbfgs,
                 mu_sgd, var_sgd)


if __name__ == "__main__":
    main()
