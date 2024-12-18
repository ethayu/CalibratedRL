import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from netcal import manual_seed
from netcal.regression import IsotonicRegression, GPBeta

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

    def probabilistic_forward(self, x, num_samples=300, calibration=None, seed=42):
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
        # create pdf from samples - probability of each sample is ratio of times it appears in the samples
        #plot pdf
        import matplotlib.pyplot as plt
        plt.hist(outputs.squeeze(), bins=30, density=True, alpha=0.6, label='Histogram (PDF Approx)')
        # kde = gaussian_kde(outputs)
        # x_vals = np.linspace(min(outputs), max(outputs), 100)
        # plt.plot(x_vals, kde(x_vals), label='Smoothed KDE', linewidth=2)

        # plt.xlabel('State')
        # plt.ylabel('Density')
        # plt.title('PDF of Next States')
        # plt.legend()
        # plt.show()
        plt.show()
        if calibration:
            # convert outputs to probability distribution]
            y = outputs.squeeze()
            y_std = outputs.std(dim=0)
            X, counts = torch.unique(outputs.squeeze(), return_counts=True)
            X = X.unsqueeze(0)
            # Step 3: Normalize counts to create probabilities
            Y = counts.float() / counts.sum()
            if calibration == 'isotonic':
                with manual_seed(seed):
                    isotonic_calibration = IsotonicRegression()
                    print(np.finfo(float).eps * torch.ones_like(X))
                    print(X)
                    isotonic_calibration.fit(outputs.squeeze().unsqueeze(0), )
                    t_iso, s_iso, q_iso = isotonic_calibration.transform((X,  torch.ones_like(X)))
                    plt.plot(s_iso.squeeze())
                    plt.show()
                    isotonic_calibration.sa
                    exit(0)
            elif calibration == 'gp_beta':
                with manual_seed(seed):
                    beta_calibration = GPBeta(n_inducing_points=12, n_random_samples=256, n_epochs=100, use_cuda=True)
                    beta_calibration.fit(outputs)
                    outputs = beta_calibration(outputs)
            else:
                raise ValueError(f"Invalid calibration method '{calibration}'.")
        return outputs.squeeze()
    
    def sample_distribution(self, distribution, num_samples=300):
        """
        Sample from the predicted distribution using Monte Carlo dropout.

        Args:
            x: Input tensor.
            num_samples: Number of Monte Carlo samples.

        Returns:
            mean: Mean of the predicted distribution.
            std: Standard deviation of the predicted distribution.
        """
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