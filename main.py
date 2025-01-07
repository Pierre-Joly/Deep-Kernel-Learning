import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Charger les données
california = fetch_california_housing()
X = california.data
y = california.target

# Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalisation des cibles
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Conversion en tenseurs PyTorch et déplacement sur l'appareil
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Affichage des dimensions
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

training_iterations = 10

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

def train_nn(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")
    
    return train_losses, test_losses

nn_model = SimpleNN(input_dim=X_train.shape[1])
train_losses, test_losses = train_nn(nn_model, X_train, y_train, X_test, y_test)

# On utilise tous les points pour le GP
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_model = ExactGPModel(X_train, y_train, likelihood)

# On place le modèle en mode entraînement
gp_model.train()
likelihood.train()

# Optimiseur
optimizer = torch.optim.AdamW([
    {'params': gp_model.parameters()},
], lr=0.1)

# "Loss" pour GPs
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = gp_model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}')

# Définir le likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
feature_extractor = SimpleNN(input_dim=X_train.shape[1])

# Le modèle DKL
dkl_model = DKLModel(X_train, y_train, likelihood, feature_extractor)

# Mettre le modèle en mode entraînement
dkl_model.train()
likelihood.train()

# Optimiseur
optimizer = torch.optim.AdamW([
    {'params': dkl_model.feature_extractor.parameters(), 'lr': 0.01},
    {'params': dkl_model.covar_module.parameters(), 'lr': 0.1},
    {'params': dkl_model.mean_module.parameters(), 'lr': 0.1},
    {'params': likelihood.parameters(), 'lr': 0.1},
])

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, dkl_model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = dkl_model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}')

nn_model.eval()
with torch.no_grad():
    predictions = nn_model(X_test)
    mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    print(f'NN Test MSE: {mse:.4f}')

gp_model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp_model(X_test))
    mse = mean_squared_error(y_test.numpy(), observed_pred.mean.numpy())
    print(f'GP Test MSE: {mse:.4f}')

dkl_model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(dkl_model(X_test))
    mse = mean_squared_error(y_test.numpy(), observed_pred.mean.numpy())
    print(f'DKL Test MSE: {mse:.4f}')

# Par exemple, courbes de perte pour le NN
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

