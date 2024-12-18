import random
import numpy as np
import torch
def enforce_reproducibility(seed=42):
    """
    Enforce reproducibility by setting random seeds in Python, NumPy, and PyTorch.

    Args:
        seed: Random seed to use for reproducibility.
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # For CUDA GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)