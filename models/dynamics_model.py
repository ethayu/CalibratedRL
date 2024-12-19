import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from netcal import manual_seed
from netcal.regression import IsotonicRegression, GPBeta
from typing import Union

class BayesianNet(nn.Module):
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
        super(BayesianNet, self).__init__()
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

    def probabilistic_forward(self, x, num_samples=300, calibrator: Union[GPBeta, IsotonicRegression] = None, seed=42):
        """
        Compute the probabilistic forward pass with Monte Carlo sampling.

        Args:
            x: Input tensor.
            num_samples: Number of Monte Carlo samples.

        Returns:
            samples: Tensor of shape (num_samples, output_dim).
        """
        self.train()  # Enable dropout for sampling
        outputs = torch.stack([self.forward(x) for _ in range(num_samples)]).detach()
        self.eval()  # Return to evaluation mode
        if calibrator:
            t, s, q = calibrator.transform(outputs.squeeze().unsqueeze(0).cpu().numpy(), 2)
            print(t.shape, t)
            print(s.shape, s)
            print(q.shape, q)
            input()
            return t.squeeze(), s.squeeze()
        else:
            return outputs
    
    def sample_distribution(self, distribution, num_samples=300, pdf=False):
        """
        Sample from the predicted distribution using Monte Carlo dropout.

        Args:
            x: Input tensor.
            num_samples: Number of Monte Carlo samples.

        Returns:
            mean: Mean of the predicted distribution.
            std: Standard deviation of the predicted distribution.
        """
        if pdf:
            points, pdf = distribution
            pdf /= np.sum(pdf)
            
            # Softmax
            exp_pdf = np.exp(pdf)  
            pdf_normalized = exp_pdf / np.sum(exp_pdf) 
            
            sampled_indices = np.random.choice(len(points), size=num_samples, p=pdf_normalized)

            # Map the sampled indices back to the corresponding points
            samples = points[sampled_indices]
            return samples.mean(), samples.std()
        else:
            sampled_values = distribution[torch.randint(0, distribution.size(0), (num_samples,))].detach().numpy()
            return sampled_values.mean(), sampled_values.std()
    
    def update_state(self, current_state, actual_demand):
        current_state = current_state.clone().detach().cpu().numpy()
        # Update rolling 7-day demand
        current_state[0] = current_state[0] * 6 / 7 + actual_demand / 7
        # Update rolling 14-day demand
        current_state[1] = current_state[1] * 13 / 14 + actual_demand / 14
        # Update rolling 28-day demand
        current_state[2] = current_state[2] * 27 / 28 + actual_demand / 28
        # Update day of the week
        current_state[3] = (current_state[3] + 1) % 7
        # Update week of the year
        if current_state[3] == 0:
            current_state[4] += 1
            current_state[4] %= 52
        # Update sine and cosine features
        current_state[5] = np.sin(2 * np.pi * (current_state[4] * 7 + current_state[3]) / 365)  
        current_state[6] = np.cos(2 * np.pi * (current_state[4] * 7 + current_state[3]) / 365)

        current_state = torch.tensor(current_state).float()
        return current_state