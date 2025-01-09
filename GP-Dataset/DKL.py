import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_data import square_exponential_kernel

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(DKLModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=feature_extractor.net[-1].out_features))
        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self, X_train, y_train, epochs=100):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {loss.item():.4f}")

if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    N = 400
    x_min, x_max = 0, 10
    
    # Uniform Grid
    X = np.linspace(x_min, x_max, N)
    
    # Kernel and Mean Functions
    length_scale = 1.0
    sigma_f = 1.0
    sigma_n = 0.1
    kernel_func = lambda x1, x2: square_exponential_kernel(x1, x2, length_scale, sigma_f, sigma_n)
    mean_func = lambda x: np.zeros(x.shape[0])
    
    # Simulate GP
    Y = np.random.multivariate_normal(mean_func(X), kernel_func(X, X))
    
    # Train-Test Split
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.5)
    
    # Convert to PyTorch Tensors
    X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
    Y_train = torch.from_numpy(Y_train).float().squeeze()
    
    # Initialize NN
    feature_extractor = SimpleNN()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLModel(X_train, Y_train, likelihood, feature_extractor)
    
    # Train DKL
    model.fit(X_train, Y_train)
    
    # Predict
    model.eval()
    likelihood.eval()
    X = torch.from_numpy(X).float().unsqueeze(-1)
    with torch.no_grad():
        Y_pred = likelihood(model(X))
    
    # Get Predictions
    mean = Y_pred.mean.detach().numpy()
    lower, upper = Y_pred.confidence_region()
    lower = lower.detach().numpy()
    upper = upper.detach().numpy()

    # Sort
    idx = np.argsort(X.numpy().squeeze())
    X = X.numpy().squeeze()[idx]
    mean = mean[idx]
    lower = lower[idx]
    upper = upper[idx]

    # Plot
    X_train = X_train.numpy().squeeze()
    Y_train = Y_train.numpy()
    plt.figure()
    plt.scatter(X_train, Y_train, color='blue', label='Train')
    plt.plot(X, mean, color='green', label='Predictions')
    plt.fill_between(X, lower, upper, color='green', alpha=0.3, label='95% CI')
    plt.legend()
    plt.show()
   
