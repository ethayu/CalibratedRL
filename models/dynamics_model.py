import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianDenseNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=5, dropout_rate=0.5):
        """
        Bayesian DenseNet with dropout for uncertainty estimation.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            hidden_dim: Number of units in each hidden layer (default: 128).
            num_layers: Number of hidden layers (default: 5).
            dropout_rate: Dropout rate for Bayesian approximation (default: 0.5).
        """
        super(BayesianDenseNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_features, hidden_dim))
            self.layers.append(nn.ReLU())  
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Dropout is active only in training mode
        return self.output_layer(x)

    def probabilistic_forward(self, x, num_samples=300):
        """
        Compute the probabilistic forward pass with Monte Carlo sampling.

        Args:
            x: Input tensor.
            num_samples: Number of Monte Carlo samples.

        Returns:
            mean: Mean prediction across samples.
            std: Standard deviation of predictions across samples.
        """
        self.train()  # Enable dropout for sampling
        outputs = torch.stack([self.forward(x) for _ in range(num_samples)])
        self.eval()  # Return to evaluation mode
        return outputs.mean(dim=0), outputs.std(dim=0)