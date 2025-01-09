import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from sklearn.metrics import accuracy_score

# Prétraitement des données MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Définition du réseau de base
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10 classes pour MNIST
        )
        
    def forward(self, x):
        return self.net(x)

# Deep Kernel Learning Model
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

# Fonction pour entraîner un réseau NN
def train_nn(model, loader, criterion, optimizer, epochs=5):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(loader))
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {total_loss / len(loader):.4f}")
    return train_losses

# Fonction d'évaluation pour le modèle NN
def evaluate_nn(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Initialisation des modèles
feature_extractor = SimpleNN()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=10, mixing_weights=False)
dkl_model = DKLModel(torch.rand((10, 28 * 28)), torch.randint(0, 10, (10,)), likelihood, feature_extractor)

# Optimiseur et critère
nn_model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer_nn = optim.Adam(nn_model.parameters(), lr=0.001)

# Entraînement du réseau de base
train_losses = train_nn(nn_model, train_loader, criterion, optimizer_nn, epochs=5)

# Évaluation
accuracy = evaluate_nn(nn_model, test_loader)
print(f"NN Test Accuracy: {accuracy:.4f}")

# Passage à DKL
dkl_model.train()
likelihood.train()

# Optimiseur pour DKL
optimizer_dkl = optim.AdamW([
    {'params': dkl_model.feature_extractor.parameters(), 'lr': 0.01},
    {'params': dkl_model.covar_module.parameters(), 'lr': 0.1},
    {'params': dkl_model.mean_module.parameters(), 'lr': 0.1},
    {'params': likelihood.parameters(), 'lr': 0.1},
])

# Affichage de la courbe de perte
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
