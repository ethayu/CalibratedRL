from netcal import manual_seed
from netcal.regression import IsotonicRegression, GPBeta
import pandas as pd
from training import enforce_reproducibility
import torch
from models.dynamics_model import BayesianNet
import os
import numpy as np

def main():
    # Define the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Paths
    train_csv = "data/train_processed.csv"
    model_dir = "models/item_models"
    
    # Enforce reproducibility
    enforce_reproducibility()

    # Load test data
    test_data = pd.read_csv(train_csv)

    # Get unique items
    unique_items = test_data['item_nbr'].unique()
    
    for item_nbr in unique_items:
        print(f"Evaluating item: {item_nbr}")

        # Filter test data for this item
        item_data = test_data[test_data['item_nbr'] == item_nbr]
        state_dim = len(item_data.columns) - 3  # Excluding item_nbr, date, and unit_sales
        
        model = BayesianNet(input_dim=state_dim, output_dim=1).to(device)
        model_path = os.path.join(model_dir, f"model_item_{item_nbr}.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        gts = []
        preds = []
        for row in item_data.iterrows():
            state = torch.tensor(row[1].drop(labels=["item_nbr", "date", "unit_sales"]).values.astype(np.float32), dtype=torch.float32, device=device)
            target = row[1]['unit_sales']
            
            samples = model.probabilistic_forward(state, num_samples=300, calibrator=None)
            gts.append(target)
            preds.append(samples)
            
    isotonic = IsotonicRegression()
    gp_beta = GPBeta()
    
    preds = np.array(preds).T
    gts = np.array(gts)
    
    idx = np.random.permutation(len(preds))
    preds = preds[:, idx]
    gts = gts[idx]
    
    print(preds.shape, gts.shape)
    
    isotonic.fit(preds, gts)
    gp_beta.fit(preds, gts)
    
    os.makedirs("models/calibration", exist_ok=True)
    
    isotonic.save_model(os.path.join("models/calibration", 'isotonic.pkl'))
    gp_beta.save_model(os.path.join("models/calibration", 'gp_beta.pkl'))
            
if __name__ == "__main__":
    main()
