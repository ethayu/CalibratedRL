from netcal import manual_seed
from netcal.regression import IsotonicRegression, GPBeta
import pandas as pd
from training import enforce_reproducibility

def main():
    # Enforce reproducibility
    enforce_reproducibility()

    # Load test data
    test_data = pd.read_csv()

    # Get unique items
    unique_items = test_data['item_nbr'].unique()


if __name__ == "__main__":
    main()
