import torch

class HeuristicPlanner:
    def __init__(self, model, safety_factor=1.5, num_samples=300, device="cpu", calibration=None):
        """
        Heuristic planner that sets inventory to a safety factor * expected demand.

        Args:
            model: Trained BayesianNet transition model.
            safety_factor: Multiplier for expected demand to avoid stock-outs.
            num_samples: Number of Monte Carlo samples for Bayesian inference.
            device: Device to run computations on (e.g., "cpu", "cuda", or "mps").
        """
        self.model = model.to(device)
        self.safety_factor = safety_factor
        self.num_samples = num_samples
        self.device = device
        self.calibration = calibration

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
        state = state[1:]

        # Predict demand for the next day
        input_data = state.unsqueeze(0)  # Add batch dimension
        samples = self.model.probabilistic_forward(input_data, num_samples=self.num_samples, calibration=self.calibration)
        mean = samples.mean(dim=0)
        expected_demand = mean.squeeze().item()

        # Set inventory to safety_factor * expected_demand
        target_inventory = self.safety_factor * expected_demand

        action = int(target_inventory)  # Avoid negative orders
        return int(action)