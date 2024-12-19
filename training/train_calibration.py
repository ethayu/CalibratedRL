import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from netcal.regression import IsotonicRegression, GPBeta
from training import enforce_reproducibility
from models.dynamics_model import BayesianNet
import os
import numpy as np

class InventoryDataset(Dataset):
    def __init__(self, csv_data, item_nbr, device="cpu"):
        self.data = csv_data[csv_data['item_nbr'] == item_nbr]
        self.features = self.data.drop(columns=['date', 'item_nbr', 'unit_sales']).values
        self.targets = self.data['unit_sales'].values
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.targets[idx], dtype=torch.float32, device=self.device)
        return x, y

def main():
    # Define the device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Paths
    train_csv = "data/train_processed.csv"
    model_dir = "models/item_models"

    # Enforce reproducibility
    enforce_reproducibility()

    # Load train data
    train_data = pd.read_csv(train_csv)

    # Get unique items
    unique_items = train_data['item_nbr'].unique()
    import time
    start = time.time()
    for item_nbr in unique_items:
        print(f"Evaluating item: {item_nbr}")

        # Create Dataset and DataLoader
        dataset = InventoryDataset(train_data, item_nbr, device=device)
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)

        state_dim = dataset.features.shape[1]
        model = BayesianNet(input_dim=state_dim, output_dim=1).to(device)
        model_path = os.path.join(model_dir, f"model_item_{item_nbr}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        total_rows = len(dataset)
        gts = np.empty(total_rows, dtype=np.float32) 
        preds = np.empty((300, total_rows), dtype=np.float32) 
        row_idx = 0
        with torch.no_grad():
            for batch_idx, (states, targets) in enumerate(dataloader):
                # Get probabilistic samples
                samples = model.probabilistic_forward(states, num_samples=300, calibrator=None).squeeze(-1)

                batch_size = len(targets)

                # Store ground truths
                gts[row_idx:row_idx + batch_size] = targets.cpu().numpy()
                
                # Store predictions
                preds[:, row_idx:row_idx + batch_size] = samples.cpu().numpy()

                row_idx += batch_size
        idx = np.random.permutation(len(gts))
        preds = preds[:, idx]
        gts = gts[idx]

        # Calibration models
        isotonic = IsotonicRegression()
        gp_beta = GPBeta(n_epochs=20, use_cuda=True)

        isotonic.fit(preds, gts)
        gp_beta.fit(preds, gts)

        # Save calibration models
        os.makedirs("models/calibrators", exist_ok=True)
        isotonic.save_model(os.path.join("models/calibrators", f'isotonic_item_{item_nbr}.pkl'))
        gp_beta.save_model(os.path.join("models/calibrators", f'gp_beta_item_{item_nbr}.pkl'))
        print(f"time taken for item {item_nbr}: {time.time() - start}")
        start = time.time()
if __name__ == "__main__":
    main()
