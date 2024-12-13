import pandas as pd
import torch
from torch.utils.data import Dataset

class InventoryDataset(Dataset):
    def __init__(self, csv_path, item_nbr, device="cpu"):
        """
        Dataset for a specific inventory item.

        Args:
            csv_path: Path to the CSV file.
            item_nbr: The item number for which this dataset is created.
            device: Device to allocate tensors to (e.g., "cpu", "cuda", or "mps").
        """
        # Load the CSV data
        self.data = pd.read_csv(csv_path)

        # Filter data for the specific item
        self.data = self.data[self.data['item_nbr'] == item_nbr]

        # Handle missing values (fill with 0 for rolling features)
        self.data.fillna(0, inplace=True)

        # Preserve date for debugging/evaluation
        self.dates = self.data['date']

        # Extract features and target
        self.features = self.data.drop(columns=['date', 'item_nbr', 'unit_sales']).values
        self.targets = self.data['unit_sales'].values

        # Store device
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieve a single item from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Tuple: (features, target).
        """
        # Convert features and target to tensors
        x = torch.tensor(self.features[idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.targets[idx], dtype=torch.float32, device=self.device)
        return x, y