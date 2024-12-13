import pandas as pd
import torch
import os
from models.mpc_planning import InventoryMPC
from models.heuristic_planning import HeuristicPlanner
from models.dynamics_model import BayesianDenseNet

def evaluate_planner(planner, data, model, device):
    """
    Evaluate a planner on a dataset.

    Args:
        planner: Planning model (MPC or HeuristicPlanner).
        data: DataFrame of test data with state features.
        model: Trained transition model.
        device: Device to run computations on (e.g., "cpu", "cuda", or "mps").

    Returns:
        Evaluation results: total shipped, wasted, stock-outs, % waste, % stock-outs, reward.
    """
    shipped = 0
    wasted = 0
    stockouts = 0
    total_reward = 0

    for _, row in data.iterrows():
        # Get the current state
        state = torch.tensor(row.drop(columns=["item_nbr", "date"]).values, dtype=torch.float32, device=device)

        # Plan the next action
        action = planner.plan(state)

        # Simulate transition
        input_data = torch.cat([state, torch.tensor([action], dtype=torch.float32, device=device)])
        input_data = input_data.unsqueeze(0)  # Add batch dimension
        mean, _ = model.probabilistic_forward(input_data, num_samples=300)
        next_state = mean.squeeze().detach()

        # Compute waste and stockouts
        inventory_level = next_state.sum().item()
        demand = next_state[-1].item()
        waste = max(0, inventory_level - demand)
        stockout = max(0, demand - inventory_level)

        # Update metrics
        shipped += action
        wasted += waste
        stockouts += stockout
        reward = -(waste + stockout)  # Negative for waste and stockouts
        total_reward += reward

    # Compute percentages
    percent_waste = (wasted / shipped) * 100 if shipped > 0 else 0
    percent_stockouts = (stockouts / shipped) * 100 if shipped > 0 else 0

    return {
        "Shipped": shipped,
        "Wasted": wasted,
        "Stockouts": stockouts,
        "% Waste": percent_waste,
        "% Stockouts": percent_stockouts,
        "Reward": total_reward
    }

def main():
    # Define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Paths
    test_csv = "data/test_processed.csv"
    model_dir = "models/item_models"

    # Load test data
    test_data = pd.read_csv(test_csv)

    # Get unique items
    unique_items = test_data['item_nbr'].unique()

    # Initialize results
    mpc_results_all = []
    heuristic_results_all = []

    # Evaluate each item separately
    for item_nbr in unique_items:
        print(f"Evaluating item: {item_nbr}")

        # Filter test data for this item
        item_data = test_data[test_data['item_nbr'] == item_nbr]

        # Load the model for this item
        model = BayesianDenseNet(input_dim=7, output_dim=1).to(device)
        model_path = os.path.join(model_dir, f"model_item_{item_nbr}.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Initialize planners
        mpc = InventoryMPC(model, input_dim=7, device=device)
        heuristic = HeuristicPlanner(model, device=device)

        # Evaluate planners
        mpc_results = evaluate_planner(mpc, item_data, model, device)
        heuristic_results = evaluate_planner(heuristic, item_data, model, device)

        # Store results
        mpc_results['item_nbr'] = item_nbr
        heuristic_results['item_nbr'] = item_nbr
        mpc_results_all.append(mpc_results)
        heuristic_results_all.append(heuristic_results)

    # Aggregate results
    mpc_summary = pd.DataFrame(mpc_results_all).mean()
    heuristic_summary = pd.DataFrame(heuristic_results_all).mean()

    # Print results
    print("\nMPC Summary Results:")
    print(mpc_summary)
    print("\nHeuristic Summary Results:")
    print(heuristic_summary)

if __name__ == "__main__":
    main()
