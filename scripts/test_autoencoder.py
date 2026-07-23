from pathlib import Path
import sys

Project_Root = Path(__file__).resolve().parents[1]
sys.path.append(str(Project_Root))

import torch
from src.models.autoencoder import AutoEncoder

batch = torch.randn(8, 3, 128, 128)
model = AutoEncoder()
output = model(batch)

print("Input shape:", batch.shape)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print("Output min:", output.min().item())
print("Output max:", output.max().item())