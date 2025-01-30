from autograd import grad
import numpy as np
from autograd import numpy as anp 
from scipy.optimize import minimize
from model import cov_matrix, unpack_params, pack_params, init_params

def negative_log_likelihood(params, X, y):
    K = cov_matrix(params, X)  # (N, N)
    N = X.shape[0]
    
    # Cholesky decomposition
    L = anp.linalg.cholesky(K)
    alpha = anp.linalg.solve(L.T, anp.linalg.solve(L, y))
    
    # y^T K^-1 y
    term1 = 0.5 * anp.dot(y.T, alpha)
    # log|K| = 2 * somme(log(diag(L)))
    term2 = 2.0 * anp.sum(anp.log(anp.diag(L)))
    # 0.5 log|K|
    term2 = 0.5 * term2
    
    # 0.5 * N * log(2*pi)
    term3 = 0.5 * N * anp.log(2.0 * anp.pi)
    
    return (term1 + term2 + term3).ravel()[0]

def objective_and_grad(theta, X, y, input_dim, feature_dim):
    # Déballer theta en params_dict
    params_dict = unpack_params(theta, input_dim, feature_dim)
    
    # Calcul de la perte
    loss_value = negative_log_likelihood(params_dict, X, y)
    
    # Calcul des gradients avec autograd
    grad_func = grad(negative_log_likelihood)
    grad_value = grad_func(params_dict, X, y)
    
    # Emballer les gradients
    grad_vector = pack_params(grad_value)
    
    return loss_value, grad_vector

def reduce_lr_on_plateau(current_lr, loss_history, factor=0.5, patience=10, min_lr=1e-6):
    """
    Réduit le learning rate si la perte stagne.
    
    Args:
        current_lr (float): Taux d'apprentissage actuel.
        loss_history (list): Historique des pertes.
        factor (float): Facteur de réduction du learning rate.
        patience (int): Nombre d'epochs avant réduction.
        min_lr (float): Limite inférieure du learning rate.
    
    Returns:
        float: Nouveau taux d'apprentissage.
    """
    if len(loss_history) > patience:
        recent_losses = loss_history[-patience:]
        if all(loss >= min(recent_losses) for loss in recent_losses):
            new_lr = max(current_lr * factor, min_lr)
            print(f"Réduction du learning rate : {current_lr:.1e} -> {new_lr:.1e}")
            return new_lr
    return current_lr

def train_gradient_descent_with_trace(
    X, y, params_dict, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, 
    n_epochs=500, patience=10, factor=0.5, min_lr=1e-6
):
    """
    Entraîne les paramètres en utilisant l'optimiseur AdamW avec un scheduler basé
    sur la stagnation de la perte.
    
    Args:
        X (np.ndarray): Données d'entrée.
        y (np.ndarray): Cibles.
        params_dict (dict): Dictionnaire des paramètres à optimiser.
        lr (float): Taux d'apprentissage initial.
        betas (tuple): Coefficients de moments (beta1, beta2).
        eps (float): Terme de stabilité numérique.
        weight_decay (float): Facteur de décroissance de poids.
        n_epochs (int): Nombre d'epochs.
        patience (int): Nombre d'epochs avant réduction du learning rate.
        factor (float): Facteur de réduction du learning rate.
        min_lr (float): Learning rate minimal.
    
    Returns:
        final_params (dict): Paramètres optimisés.
        traj_params (list): Liste des vecteurs de paramètres à chaque epoch.
    """
    trajectory = []
    loss_history = []  # Historique des pertes pour le scheduler
    grad_func = grad(negative_log_likelihood)

    theta_vec = pack_params(params_dict)
    trajectory.append(theta_vec)
    
    # Initialiser les moments m et v
    m = {k: np.zeros_like(v) for k, v in params_dict.items()}
    v = {k: np.zeros_like(v) for k, v in params_dict.items()}
    
    beta1, beta2 = betas
    for epoch in range(1, n_epochs + 1):
        # Calcul de la perte et des gradients
        loss_value = negative_log_likelihood(params_dict, X, y)
        loss_history.append(loss_value)
        grads_dict = grad_func(params_dict, X, y)
        
        # Mettre à jour les moments
        for k in params_dict.keys():
            m[k] = beta1 * m[k] + (1 - beta1) * grads_dict[k]
            v[k] = beta2 * v[k] + (1 - beta2) * (grads_dict[k] ** 2)
        
        # Correction de biais
        m_hat = {k: m[k] / (1 - beta1 ** epoch) for k in params_dict.keys()}
        v_hat = {k: v[k] / (1 - beta2 ** epoch) for k in params_dict.keys()}
        
        # Mettre à jour les paramètres avec AdamW
        for k in params_dict.keys():
            params_dict[k] -= lr * (m_hat[k] / (np.sqrt(v_hat[k]) + eps) + weight_decay * params_dict[k])
        
        # Enregistrer la trajectoire
        theta_vec = pack_params(params_dict)
        trajectory.append(theta_vec)
        
        # Scheduler : Réduire le learning rate si nécessaire
        if lr > min_lr:
            lr = reduce_lr_on_plateau(current_lr=lr, loss_history=loss_history, factor=factor, patience=patience, min_lr=min_lr)
        
        # Affichage périodique
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss = {loss_value:.5f}, Learning Rate = {lr:.1e}")
    
    return params_dict, trajectory


def train_lbfgsb_with_trace(X, y, theta_init, input_dim, feature_dim, maxiter=100):
    """
    Entraîne les paramètres en utilisant l'optimiseur L-BFGS-B.
    
    Args:
        X (np.ndarray): Données d'entrée.
        y (np.ndarray): Cibles.
        theta_init (np.ndarray): Vecteur initial des paramètres empaquetés.
        input_dim (int): Dimension d'entrée.
        feature_dim (int): Dimension des features.
        maxiter (int): Nombre maximal d'itérations.
    
    Returns:
        final_params (dict): Paramètres optimisés.
        trajectory (list): Liste des vecteurs de paramètres à chaque itération.
    """
    trajectory = []
    trajectory.append(theta_init.copy())
    
    def callback_f(theta):
        trajectory.append(theta.copy())
    
    res = minimize(
        fun=lambda th: objective_and_grad(th, X, y, input_dim, feature_dim),
        x0=theta_init,
        method='L-BFGS-B',
        jac=True,
        callback=callback_f,
        options={'maxiter': maxiter, 'disp': False}
    )
    
    final_params = unpack_params(res.x, input_dim, feature_dim)
    return final_params, trajectory

