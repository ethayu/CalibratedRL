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

assert len(dataframes) == 6, "Incorrect number of folders in the results directory"
assert all(isinstance(df, pd.DataFrame) for df in dataframes.values()) and all(len(df) == 62 for df in dataframes.values()), "Dataframes not created correctly"

# Example: Print the dataframe for a specific folder
for folder, df in dataframes.items():
    print(f"DataFrame for folder {folder}:")
    print(df)