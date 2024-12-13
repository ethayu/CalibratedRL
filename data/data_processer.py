import pandas as pd
import numpy as np
import os

# Get the current working directory
cwd = os.getcwd()

# Load data
data = pd.read_csv(f"{cwd}/data/train.csv", usecols=['date', 'store_nbr', 'item_nbr', 'unit_sales'])
data['date'] = pd.to_datetime(data['date'])

# Print the number of rows
print(f"Number of rows: {len(data)}")

# Treat negative sales as returns
data.loc[:, 'unit_sales'] = data['unit_sales']

# Find the top 100 stores by total sales
top_stores = data.groupby('store_nbr')['unit_sales'].sum().nlargest(100).index

# Filter data for the top 100 stores
data = data[data['store_nbr'].isin(top_stores)]

# Find the top 100 items by total sales within the top 100 stores
top_items = data.groupby('item_nbr')['unit_sales'].sum().nlargest(100).index

# Filter data for the top 100 items
data = data[data['item_nbr'].isin(top_items)]

# Drop the `store_nbr` column as it's no longer needed
data = data.drop(columns=['store_nbr'])

# Sort by date for time series
data = data.sort_values('date')

# Generate rolling means
data['rolling_7'] = data.groupby('item_nbr')['unit_sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
data['rolling_14'] = data.groupby('item_nbr')['unit_sales'].transform(lambda x: x.rolling(14, min_periods=1).mean())
data['rolling_28'] = data.groupby('item_nbr')['unit_sales'].transform(lambda x: x.rolling(28, min_periods=1).mean())

# Add day of the week and week of the year
data['day_of_week'] = data['date'].dt.dayofweek
data['week_of_year'] = data['date'].dt.isocalendar().week

# Add sine and cosine features for seasonal patterns
data['day_sin'] = np.sin(2 * np.pi * data['date'].dt.dayofyear / 365)
data['day_cos'] = np.cos(2 * np.pi * data['date'].dt.dayofyear / 365)

# Split into training and testing data
train_data = data[data['date'] < "2017-06-01"]
test_data = data[data['date'] >= "2017-06-01"]

# Save the processed data
os.makedirs(f"{cwd}/data", exist_ok=True)
train_data.to_csv(f"{cwd}/data/train_processed.csv", index=False)
test_data.to_csv(f"{cwd}/data/test_processed.csv", index=False)

# Print statistics
print(f"Number of training samples: {len(train_data)}")
print(f"Number of testing samples: {len(test_data)}")