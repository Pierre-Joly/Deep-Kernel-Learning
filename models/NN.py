import torch
import torch.nn as nn

class NN:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        return model

    def fit(self, X_train, y_train, epochs=100):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=64,
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