import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.dynamics_model import BayesianNet
from data.inventory_dataloader import InventoryDataset

def train_item_model(item_nbr, train_csv, output_dir, device, input_dim=7, output_dim=1):
    """
    Train a separate model for a specific item.

    Args:
        item_nbr: The item number.
        train_csv: Path to the training dataset.
        output_dir: Directory to save trained models.
        device: Device to run computations on (e.g., "cpu", "cuda", or "mps").
        input_dim: Number of input features.
        output_dim: Number of output features.
    """
    # Create dataset and dataloader
    dataset = InventoryDataset(train_csv, item_nbr, device=device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = BayesianNet(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(50):  # Adjust number of epochs as needed
        total_loss = 0
        for features, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / dataloader.batch_size
        print(f"Item {item_nbr} | Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_item_{item_nbr}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for item {item_nbr} at {model_path}")

def main():
    # Paths and settings
    train_csv = "data/train_processed.csv"
    output_dir = "models/item_models"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load unique items from the dataset
    data = pd.read_csv(train_csv)
    unique_items = data['item_nbr'].unique()

    # Train a model for each item
    for item_nbr in unique_items:
        train_item_model(item_nbr, train_csv, output_dir, device)

if __name__ == "__main__":
    main()
