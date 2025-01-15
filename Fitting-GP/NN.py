import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_data import gaussian_process, create_kernel, create_mean_func
import torch
import torch.nn as nn

class NN:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(1, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )
        return model

    def fit(self, X_train, y_train, epochs=1000):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=len(X_train),
            shuffle=True
        )
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, Y_batch in data_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_loss / len(data_loader):.4f}")

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            return self.model(X_test)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    N = 400
    x_min, x_max = 0, 10

    # Uniform Grid
    X = np.linspace(x_min, x_max, N)

    # Kernel and Mean Functions
    length_scale = 1.0
    sigma_f = 1.0
    sigma_n = 0.1
    kernel_func = create_kernel(length_scale, sigma_f)
    mean_func = create_mean_func(0.0)

    # Simulate GP
    Y = gaussian_process(X, mean_func, kernel_func, sigma_n)

    # Split Data
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.5)

    # Prepare Tensors
    X_train = torch.tensor(X_train).float().unsqueeze(-1)
    Y_train = torch.tensor(Y_train).float().unsqueeze(-1)
    X = torch.tensor(X).float().unsqueeze(-1)

    # Train NN
    NN_model = NN()
    NN_model.fit(X_train, Y_train)

    # Predict
    Y_pred = NN_model.predict(X).detach()

    # Convert to NumPy for Plotting
    X_train = X_train.numpy().squeeze()
    Y_train = Y_train.numpy().squeeze()
    X = X.numpy().squeeze()
    Y_pred = Y_pred.numpy().squeeze()

    # Sort X for Plotting
    idx = np.argsort(X)
    X = X[idx]
    Y_pred = Y_pred[idx]

    # Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(X_train, Y_train, 'ro', label='Training Data')
    plt.plot(X, Y_pred, label='NN Prediction')
    plt.title('NN Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
