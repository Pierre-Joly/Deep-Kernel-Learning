import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
from model import init_params, pack_params, unpack_params
from optimization import negative_log_likelihood
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

def sample_param_space(n_samples,
                       input_dim=1, feature_dim=2,
                       range_min=-3.0, range_max=3.0,
                       seed=42):
    rng = np.random.RandomState(seed)
    
    # Get dimensions
    dummy = init_params(input_dim, feature_dim)
    dummy_vec = pack_params(dummy)
    param_dim = dummy_vec.shape[0]
    
    # Uniform sampling
    param_samples = rng.uniform(range_min, range_max, size=(n_samples, param_dim))
    
    return param_samples


def compute_nll_for_param_set(X, y, param_set, input_dim=1, feature_dim=2):
    n_samples = param_set.shape[0]
    nll_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        theta_vec = param_set[i]
        params_dict = unpack_params(theta_vec, input_dim, feature_dim)
        nll_values[i] = negative_log_likelihood(params_dict, X, y)
    
    return nll_values

def umap_reduction(param_trajectories, n_components=2):
    arr = np.vstack(param_trajectories)
    
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(arr)
    return embedding

def plot_results(X_train, y_train, X_test, 
                 mu_lbfgs, var_lbfgs, 
                 mu_sgd, var_sgd):
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='red', marker='x', label='Training data')

    # LBFGS
    plt.plot(X_test, mu_lbfgs, color='blue', label='Mean prediction (LBFGS)')
    plt.fill_between(X_test.ravel(),
                     mu_lbfgs - 2.0*np.sqrt(var_lbfgs),
                     mu_lbfgs + 2.0*np.sqrt(var_lbfgs),
                     alpha=0.2, color='blue')
    
    # SGD
    plt.plot(X_test, mu_sgd, color='green', label='Mean prediction (SGD)')
    plt.fill_between(X_test.ravel(),
                     mu_sgd - 2.0*np.sqrt(var_sgd),
                     mu_sgd + 2.0*np.sqrt(var_sgd),
                     alpha=0.2, color='green')
    
    plt.title("Comparaison DKL : LBFGS vs SGD")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trajectories(param_samples, embedding_sgd, embedding_lbfgs, nll_values, reduction="pca"):

    all_points = np.vstack([param_samples, embedding_sgd, embedding_lbfgs])

    if reduction == "pca":
        # Réduction via PCA
        pca = PCA(n_components=2)
        embedding_all = pca.fit_transform(all_points)
        x_title = 'PC1'
        y_title = 'PC2'
    elif reduction == "umap":
        # Réduction via UMAP
        embedding_all = umap_reduction(all_points, n_components=2)
        x_title = 'UMAP1'
        y_title = 'UMAP2'
    else:
        raise ValueError("Unknown reduction method. Choose 'pca' or 'umap'.")

    n_samples   = len(param_samples)
    n_sgd       = len(embedding_sgd)
    n_lbfgs     = len(embedding_lbfgs)

    embedding_samples = embedding_all[:n_samples]
    embedding_sgd     = embedding_all[n_samples : n_samples + n_sgd]
    embedding_lbfgs   = embedding_all[n_samples + n_sgd : n_samples + n_sgd + n_lbfgs]

    grid_x, grid_y = np.mgrid[
        embedding_samples[:,0].min():embedding_samples[:,0].max():100j,
        embedding_samples[:,1].min():embedding_samples[:,1].max():100j
    ]

    grid_z = griddata(
        points=embedding_samples,
        values=nll_values,
        xi=(grid_x, grid_y),
        method='cubic'
    )

    fig = go.Figure()

    fig = go.Figure(data=[
        go.Contour(
            x=grid_x[:,0],
            y=grid_y[0],
            z=grid_z,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            name='NLL Contour'
        )
        ])
    
    fig.add_trace(
        go.Scatter(
            x=embedding_sgd[:,0],
            y=embedding_sgd[:,1],
            mode='lines+markers',
            marker=dict(
                size=6,
                color='red'
            ),
            line=dict(color='red'),
            name='SGD'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=embedding_lbfgs[:,0],
            y=embedding_lbfgs[:,1],
            mode='lines+markers',
            marker=dict(
                size=6,
                color='blue'
            ),
            line=dict(color='blue'),
            name='LBFGS'
        )
    )

    fig.update_layout(
        xaxis_title='Axis X',
        yaxis_title='Axis Y'
    )
    fig.show()

