import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class GaussianMF(nn.Module):
    def __init__(self, num_mfs, input_dim):
        super(GaussianMF, self).__init__()
        self.num_mfs = num_mfs
        self.input_dim = input_dim
        self.means = nn.Parameter(torch.randn(input_dim, num_mfs))  
        self.sigmas = nn.Parameter(torch.randn(input_dim, num_mfs).abs() + 1e-6)

    def forward(self, x):
        x = x.unsqueeze(-1)
        gauss = torch.exp(-((x - self.means) ** 2) / (2 * self.sigmas ** 2))
        return gauss 


class ANFIS(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(ANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.mf_layer = GaussianMF(num_rules, input_dim)
        self.linear_layer = self.linear_layer = nn.Sequential(nn.Linear(input_dim, num_rules, 
                                                                bias=False),nn.ReLU())
        self.rule_weights = nn.Parameter(torch.randn(num_rules, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_rules))

    def forward(self, x):
        mf_out = self.mf_layer(x)  # (batch, input_dim, num_mfs)
        rule_strengths = torch.prod(mf_out, dim=1)  # (batch, num_rules)
        norm_rule_strengths = rule_strengths / (rule_strengths.sum(dim=1, keepdim=True) + 1e-6)
        linear_output = self.linear_layer(x)  # (batch, num_rules)
        rule_output = (norm_rule_strengths * linear_output).sum(dim=1) + self.bias.sum()
        return rule_output


if __name__ == '__main__':
    df = pd.read_csv('D://2024.2//ĐANCCN//ĐATN code//water-quality-research//data convert//WQI_dataset.csv')
    X = df.drop(columns=['TSI(Chl-a)']).values
    y = df['TSI(Chl-a)'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = X[:2417], X[2417:], y[:2417], y[2417:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X.shape[1]
    num_rules = 8
    model = ANFIS(input_size, num_rules)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_train_pred = model(X_batch)
            train_loss = criterion(y_train_pred, y_batch)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_val_pred = model(X_batch)
                val_loss = criterion(y_val_pred, y_batch)
                total_val_loss += val_loss.item()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(test_loader):.4f}')
  
    model.eval()
    y_pred = model(X_test).detach().numpy()
    mse = mean_squared_error(y_test.numpy(), y_pred)
    mse_train = mean_squared_error(y_train.numpy(), model(X_train).detach().numpy())
    r_train = np.sqrt(r2_score(y_train.numpy(), model(X_train).detach().numpy()))
    r = np.sqrt(r2_score(y_test.numpy(), y_pred))

    print(f'R train: {r_train:.4f}')
    print(f'R test: {r:.4f}')
    print(f'MSE train: {mse_train:.4f}')
    print(f'MSE: {mse:.4f}')
