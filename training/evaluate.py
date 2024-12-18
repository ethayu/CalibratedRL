import pandas as pd
import torch
import os
from models.mpc_planning import InventoryMPC
from models.heuristic_planning import HeuristicPlanner
from models.dynamics_model import BayesianNet
import numpy as np
from training import enforce_reproducibility

def evaluate_planner(planner, data, device):
    """
    Evaluate a planner on a dataset.

    Args:
        planner: Planning model (MPC or HeuristicPlanner).
        data: DataFrame of test data with state features.
        device: Device to run computations on (e.g., "cpu", "cuda", or "mps").

    Returns:
        Evaluation results: total shipped, wasted, stock-outs, % waste, % stock-outs, reward.
    """
    shipped = 0
    wasted = 0
    stockouts = 0
    total_reward = 0

    inventory_level = 0

    for _, row in data.iterrows():
        # Get the current state
        state = torch.tensor(row.drop(labels=["item_nbr", "date"]).values.astype(np.float32), dtype=torch.float32, device=device)

        # Plan the next action
        action = planner.plan(state)

        inventory_level += action
        
        demand = state[0]

        # Compute waste and stockouts
        waste = max(0, inventory_level - demand)
        stockout = max(0, demand - inventory_level)

        # Update metrics
        shipped += action
        wasted += waste
        stockouts += stockout
        reward = -(waste + stockout) # Negative for waste and stockouts
        total_reward += reward
        inventory_level = waste  # Update inventory level for the next step
        break

    if type(shipped) == torch.Tensor:
        shipped = shipped.item()
    if type(wasted) == torch.Tensor:
        wasted = wasted.item()
    if type(stockouts) == torch.Tensor:
        stockouts = stockouts.item()
    if type(total_reward) == torch.Tensor:
        total_reward = total_reward.item()
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
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Paths
    test_csv = "data/test_processed.csv"
    model_dir = "models/item_models"

    # Enforce reproducibility
    enforce_reproducibility()

    # Load test data
    test_data = pd.read_csv(test_csv)

    # Get unique items
    unique_items = test_data['item_nbr'].unique()

    # Initialize results
    mpc_results_all = []
    heuristic_results_all = []
    mpc_isotonic_results_all = []
    heuristic_isotonic_results_all = []
    mpc_gp_beta_results_all = []
    heuristic_gp_beta_results_all = []

    # Evaluate each item separately
    for item_nbr in unique_items:
        print(f"Evaluating item: {item_nbr}")

        # Filter test data for this item
        item_data = test_data[test_data['item_nbr'] == item_nbr]
        state_dim = len(item_data.columns) - 3  # Excluding item_nbr, date, and unit_sales

        # Load the model for this item
        model = BayesianNet(input_dim=state_dim, output_dim=1).to(device)
        model_path = os.path.join(model_dir, f"model_item_{item_nbr}.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Initialize planners
        mpc = InventoryMPC(model, input_dim=state_dim, num_trajectories=200, device=device, calibration=None)
        heuristic = HeuristicPlanner(model, device=device, calibration=None)
        mpc_isotonic = InventoryMPC(model, input_dim=state_dim, num_trajectories=200, device=device, calibration='isotonic')
        heuristic_isotonic = HeuristicPlanner(model, device=device, calibration='isotonic')
        mpc_gp_beta = InventoryMPC(model, input_dim=state_dim, num_trajectories=200, device=device, calibration='gp_beta')
        heuristic_gp_beta = HeuristicPlanner(model, device=device, calibration='gp_beta')

        # Evaluate planners
        # mpc_results = evaluate_planner(mpc, item_data, device)
        # heuristic_results = evaluate_planner(heuristic, item_data, device)
        mpc_isotonic_results = evaluate_planner(mpc_isotonic, item_data, device)
        heuristic_isotonic_results = evaluate_planner(heuristic_isotonic, item_data, device)
        mpc_gp_beta_results = evaluate_planner(mpc_gp_beta, item_data, device)
        heuristic_gp_beta_results = evaluate_planner(heuristic_gp_beta, item_data, device)

        # Store results
        mpc_results['item_nbr'] = item_nbr
        heuristic_results['item_nbr'] = item_nbr
        mpc_isotonic_results['item_nbr'] = item_nbr
        heuristic_isotonic_results['item_nbr'] = item_nbr
        mpc_gp_beta_results['item_nbr'] = item_nbr
        heuristic_gp_beta_results['item_nbr'] = item_nbr

        mpc_results_all.append(mpc_results)
        heuristic_results_all.append(heuristic_results)
        mpc_isotonic_results_all.append(mpc_isotonic_results)
        heuristic_isotonic_results_all.append(heuristic_isotonic_results)
        mpc_gp_beta_results_all.append(mpc_gp_beta_results)
        heuristic_gp_beta_results_all.append(heuristic_gp_beta_results)

        mpc_results = pd.DataFrame(mpc_results, index=[0])
        heuristic_results = pd.DataFrame(heuristic_results, index=[0])
        mpc_isotonic_results = pd.DataFrame(mpc_isotonic_results, index=[0])
        heuristic_isotonic_results = pd.DataFrame(heuristic_isotonic_results, index=[0])
        mpc_gp_beta_results = pd.DataFrame(mpc_gp_beta_results, index=[0])
        heuristic_gp_beta_results = pd.DataFrame(heuristic_gp_beta_results, index=[0])
        mpc_results.to_csv(f"results/mpc_by_item/{item_nbr}_results.csv", index=False)
        heuristic_results.to_csv(f"results/heuristic_by_item/{item_nbr}_results.csv", index=False)
        mpc_isotonic_results.to_csv(f"results/mpc_isotonic_by_item/{item_nbr}_results.csv", index=False)
        heuristic_isotonic_results.to_csv(f"results/heuristic_isotonic_by_item/{item_nbr}_results.csv", index=False)
        mpc_gp_beta_results.to_csv(f"results/mpc_gp_beta_by_item/{item_nbr}_results.csv", index=False)
        heuristic_gp_beta_results.to_csv(f"results/heuristic_gp_beta_by_item/{item_nbr}_results.csv", index=False)

    # Convert results to DataFrames
    mpc_results_all = pd.DataFrame(mpc_results_all)
    heuristic_results_all = pd.DataFrame(heuristic_results_all)
    mpc_isotonic_results_all = pd.DataFrame(mpc_isotonic_results_all)
    heuristic_isotonic_results_all = pd.DataFrame(heuristic_isotonic_results_all)
    mpc_gp_beta_results_all = pd.DataFrame(mpc_gp_beta_results_all)
    heuristic_gp_beta_results_all = pd.DataFrame(heuristic_gp_beta_results_all)

    # Aggregate results
    mpc_summary = mpc_results_all.sum()
    mpc_summary["% Waste"] = mpc_summary["Wasted"] / mpc_summary["Shipped"] * 100
    mpc_summary["% Stockouts"] = mpc_summary["Stockouts"] / mpc_summary["Shipped"] * 100
    heuristic_summary = heuristic_results_all.sum()
    heuristic_summary["% Waste"] = heuristic_summary["Wasted"] / heuristic_summary["Shipped"] * 100
    heuristic_summary["% Stockouts"] = heuristic_summary["Stockouts"] / heuristic_summary["Shipped"] * 100
    mpc_isotonic_summary = mpc_isotonic_results_all.sum()
    mpc_isotonic_summary["% Waste"] = mpc_isotonic_summary["Wasted"] / mpc_isotonic_summary["Shipped"] * 100
    mpc_isotonic_summary["% Stockouts"] = mpc_isotonic_summary["Stockouts"] / mpc_isotonic_summary["Shipped"] * 100
    heuristic_isotonic_summary = heuristic_isotonic_results_all.sum()
    heuristic_isotonic_summary["% Waste"] = heuristic_isotonic_summary["Wasted"] / heuristic_isotonic_summary["Shipped"] * 100
    heuristic_isotonic_summary["% Stockouts"] = heuristic_isotonic_summary["Stockouts"] / heuristic_isotonic_summary["Shipped"] * 100
    mpc_gp_beta_summary = mpc_gp_beta_results_all.sum()
    mpc_gp_beta_summary["% Waste"] = mpc_gp_beta_summary["Wasted"] / mpc_gp_beta_summary["Shipped"] * 100
    mpc_gp_beta_summary["% Stockouts"] = mpc_gp_beta_summary["Stockouts"] / mpc_gp_beta_summary["Shipped"] * 100
    heuristic_gp_beta_summary = heuristic_gp_beta_results_all.sum()
    heuristic_gp_beta_summary["% Waste"] = heuristic_gp_beta_summary["Wasted"] / heuristic_gp_beta_summary["Shipped"] * 100
    heuristic_gp_beta_summary["% Stockouts"] = heuristic_gp_beta_summary["Stockouts"] / heuristic_gp_beta_summary["Shipped"] * 100

    # Print results
    print("\nMPC Summary Results:")
    print(mpc_summary)
    print("\nHeuristic Summary Results:")
    print(heuristic_summary)
    print("\nMPC Isotonic Summary Results:")
    print(mpc_isotonic_summary)
    print("\nHeuristic Isotonic Summary Results:")
    print(heuristic_isotonic_summary)
    print("\nMPC GP Beta Summary Results:")
    print(mpc_gp_beta_summary)
    print("\nHeuristic GP Beta Summary Results:")
    print(heuristic_gp_beta_summary)

    # Save results
    mpc_results_all.to_csv("results/mpc_results.csv", index=False)
    heuristic_results_all.to_csv("results/heuristic_results.csv", index=False)
    mpc_isotonic_results_all.to_csv("results/mpc_isotonic_results.csv", index=False)
    heuristic_isotonic_results_all.to_csv("results/heuristic_isotonic_results.csv", index=False)
    mpc_gp_beta_results_all.to_csv("results/mpc_gp_beta_results.csv", index=False)
    heuristic_gp_beta_results_all.to_csv("results/heuristic_gp_beta_results.csv", index=False)

if __name__ == "__main__":
    main()
