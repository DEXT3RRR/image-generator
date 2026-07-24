import torch
from torch.utils.data import DataLoader
from pathlib import Path 
import sys 
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.autoencoder import AutoEncoder
from src.data.dataset import InteriorDataset

DATA_DIR = "data/processed/interior_128/test"
OUTPUT_DIR = Path("outputs/reconstructions")
BATCH_SIZE = 8 

if torch.backends.mps.is_available():
    device = torch.device("mps")
else :
    device = torch.device("cpu")
print("Using device: ", device)

dataset = InteriorDataset(DATA_DIR)
dataloader = DataLoader (
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

model = AutoEncoder().to(device)
model.load_state_dict(torch.load("checkpoints/autoencoder_epoch_1.pth", map_location = device, weights_only=True))
model.eval()



batch = next(iter(dataloader)).to(device)
with torch.no_grad():
    output = model(batch)

print("Original batch shape:", batch.shape)
print("Reconstructed batch shape:", output.shape)

batch = batch.cpu()
output = output.cpu()
comparison = torch.cat([batch, output], dim=0)

save_image(batch, OUTPUT_DIR / "original.png", nrow=4)
save_image(output, OUTPUT_DIR / "reconstructed.png", nrow=4)
save_image(comparison, OUTPUT_DIR / "comparison.png", nrow=4)

print("Saved original images to:", OUTPUT_DIR / "original.png")
print("Saved reconstructed images to:", OUTPUT_DIR / "reconstructed.png")
print("Saved comparison image to:", OUTPUT_DIR / "comparison.png")