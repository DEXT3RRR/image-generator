from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from src.data.dataset import InteriorDataset

dataset = InteriorDataset("data/processed/interior_128/train")

dataloader = DataLoader(
    dataset = dataset,
    batch_size = 8,
    shuffle = True, 
)

batch = next(iter(dataloader))

print("Batch shape: ", batch.shape)
print("Batch dtype: ", batch.dtype)
print("Batch Max: ", batch.max().item())
print("Batch Min: ", batch.min().item())

