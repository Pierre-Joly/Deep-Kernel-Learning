import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel

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
    
    def fit(self, X_train, Y_train, epochs=100):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(X_train)
            loss = -mll(output, Y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {loss.item():.4f}")
