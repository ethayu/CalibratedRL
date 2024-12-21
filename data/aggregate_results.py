import os
import pandas as pd

# Path to the results directory
results_dir = 'results'

# Dictionary to hold dataframes for each folder
dataframes = {}

# Iterate through each folder in the results directory
for folder_name in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder_name)
    if os.path.isdir(folder_path):
        # Create an empty dataframe for the folder
        dataframes[folder_name] = pd.DataFrame()
        
        # Iterate through each CSV file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                # Read the CSV file and append it as a row to the dataframe
                df = pd.read_csv(file_path)
                dataframes[folder_name] = pd.concat([dataframes[folder_name], df], ignore_index=True)

mpc_results_all = dataframes["mpc_by_item"].drop(columns=["item_nbr"])
heuristic_results_all = dataframes["heuristic_by_item"].drop(columns=["item_nbr"])
mpc_isotonic_results_all = dataframes["mpc_isotonic_by_item"].drop(columns=["item_nbr"])
heuristic_isotonic_results_all = dataframes["heuristic_isotonic_by_item"].drop(columns=["item_nbr"])
mpc_gp_beta_results_all = dataframes["mpc_gp_beta_by_item"].drop(columns=["item_nbr"])
heuristic_gp_beta_results_all = dataframes["heuristic_gp_beta_by_item"].drop(columns=["item_nbr"])

mpc_summary = mpc_results_all.mean()
mpc_summary["% Waste"] = mpc_summary["Wasted"] / mpc_summary["Shipped"] * 100
mpc_summary["% Stockouts"] = mpc_summary["Stockouts"] / mpc_summary["Shipped"] * 100
heuristic_summary = heuristic_results_all.mean()
heuristic_summary["% Waste"] = heuristic_summary["Wasted"] / heuristic_summary["Shipped"] * 100
heuristic_summary["% Stockouts"] = heuristic_summary["Stockouts"] / heuristic_summary["Shipped"] * 100
mpc_isotonic_summary = mpc_isotonic_results_all.mean()
mpc_isotonic_summary["% Waste"] = mpc_isotonic_summary["Wasted"] / mpc_isotonic_summary["Shipped"] * 100
mpc_isotonic_summary["% Stockouts"] = mpc_isotonic_summary["Stockouts"] / mpc_isotonic_summary["Shipped"] * 100
heuristic_isotonic_summary = heuristic_isotonic_results_all.mean()
heuristic_isotonic_summary["% Waste"] = heuristic_isotonic_summary["Wasted"] / heuristic_isotonic_summary["Shipped"] * 100
heuristic_isotonic_summary["% Stockouts"] = heuristic_isotonic_summary["Stockouts"] / heuristic_isotonic_summary["Shipped"] * 100
mpc_gp_beta_summary = mpc_gp_beta_results_all.mean()
mpc_gp_beta_summary["% Waste"] = mpc_gp_beta_summary["Wasted"] / mpc_gp_beta_summary["Shipped"] * 100
mpc_gp_beta_summary["% Stockouts"] = mpc_gp_beta_summary["Stockouts"] / mpc_gp_beta_summary["Shipped"] * 100
heuristic_gp_beta_summary = heuristic_gp_beta_results_all.mean()
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

os.makedirs("results/summary", exist_ok=True)
mpc_summary.to_csv("results/summary/mpc.csv")
heuristic_summary.to_csv("results/summary/heuristic.csv")
mpc_isotonic_summary.to_csv("results/summary/mpc_isotonic.csv")
heuristic_isotonic_summary.to_csv("results/summary/heuristic_isotonic.csv")
mpc_gp_beta_summary.to_csv("results/summary/mpc_gp_beta.csv")
heuristic_gp_beta_summary.to_csv("results/summary/heuristic_gp_beta.csv")