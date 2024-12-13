import torch
import numpy as np

class InventoryMPC:
    def __init__(self, model, input_dim, horizon=5, num_trajectories=1000, num_samples=300, device="cpu"):
        """
        Initialize MPC for inventory management with probabilistic sampling.

        Args:
            model: Trained BayesianDenseNet transition model.
            input_dim: Number of features in the state.
            horizon: Planning horizon (number of steps).
            num_trajectories: Number of trajectories to sample.
            num_samples: Number of Monte Carlo samples for Bayesian inference.
            device: Device to run computations on (e.g., "cpu", "cuda", or "mps").
        """
        self.model = model.to(device)
        self.input_dim = input_dim
        self.horizon = horizon
        self.num_trajectories = num_trajectories
        self.num_samples = num_samples
        self.device = device

    def simulate_trajectory(self, initial_state):
        """
        Simulate a single trajectory by sampling actions from the predicted demand distribution.

        Args:
            initial_state: Initial state vector.

        Returns:
            Tuple: (cumulative reward, first action).
        """
        total_reward = 0
        current_state = initial_state.clone().to(self.device)
        first_action = None

        for step in range(self.horizon):
            # Predict the next state's demand distribution
            input_data = current_state.unsqueeze(0)  # Add batch dimension
            mean, std = self.model.probabilistic_forward(input_data, num_samples=self.num_samples)
            mean, std = mean.to(self.device), std.to(self.device)

            # Sample demand from the predicted distribution
            sampled_demand = torch.normal(mean, std).squeeze().item()

            # Determine the action (order quantity) to match the sampled demand
            action = max(0, sampled_demand - current_state.sum().item())  # Ensure non-negative actions

            # Keep the first action for returning later
            if step == 0:
                first_action = int(action)

            # Simulate the state transition
            input_data = torch.cat([current_state, torch.tensor([action], dtype=torch.float32, device=self.device)])
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            next_mean, next_std = self.model.probabilistic_forward(input_data, num_samples=self.num_samples)

            # Sample the next state from the predicted distribution
            next_state = torch.normal(next_mean, next_std).squeeze()

            # Compute waste and stock-outs
            inventory_level = next_state.sum().item()
            waste = max(0, inventory_level - sampled_demand)
            stockout = max(0, sampled_demand - inventory_level)

            # Compute reward (negative penalties for waste and stock-outs)
            reward = -(waste + stockout)
            total_reward += reward

            # Update the current state for the next step
            current_state = next_state[:-1]  # Decrement shelf life (last shelf life is spoiled)

        return total_reward, first_action

    def plan(self, initial_state):
        """
        Perform MPC by simulating trajectories and selecting the best first action.

        Args:
            initial_state: Current state vector.

        Returns:
            Best action to take in the current state.
        """
        best_action = None
        best_reward = float('-inf')

        for _ in range(self.num_trajectories):
            # Simulate a trajectory and get its reward and first action
            trajectory_reward, trajectory_action = self.simulate_trajectory(initial_state)

            # Update the best action based on the trajectory's reward
            if trajectory_reward > best_reward:
                best_reward = trajectory_reward
                best_action = trajectory_action

        return best_action