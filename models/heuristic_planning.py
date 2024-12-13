import torch

class HeuristicPlanner:
    def __init__(self, model, safety_factor=1.5, num_samples=300, device="cpu"):
        """
        Heuristic planner that sets inventory to a safety factor * expected demand.

        Args:
            model: Trained BayesianDenseNet transition model.
            safety_factor: Multiplier for expected demand to avoid stock-outs.
            num_samples: Number of Monte Carlo samples for Bayesian inference.
            device: Device to run computations on (e.g., "cpu", "cuda", or "mps").
        """
        self.model = model.to(device)
        self.safety_factor = safety_factor
        self.num_samples = num_samples
        self.device = device

    def plan(self, state):
        """
        Compute the heuristic action based on the safety factor.

        Args:
            state: Current state vector.

        Returns:
            Action (order quantity) to bring inventory to the target level.
        """
        # Move state to the appropriate device
        state = state.to(self.device)

        # Predict demand for the next day
        input_data = state.unsqueeze(0)  # Add batch dimension
        mean, _ = self.model.probabilistic_forward(input_data, num_samples=self.num_samples)
        expected_demand = mean.squeeze().item()

        # Set inventory to safety_factor * expected_demand
        target_inventory = self.safety_factor * expected_demand

        # Compute the action (order quantity)
        current_inventory = state.sum().item()  # Total inventory in current state
        action = max(0, target_inventory - current_inventory)  # Avoid negative orders
        return int(action)