import torch
from torch.utils.data import DataLoader
from src.data.dataset import InteriorDataset
from src.models.autoencoder import AutoEncoder



TRAIN_DATA = "data/processed/interior_128/train"
BATCH_SIZE = 8 
LEARNING_RATE = 0.001
EPOCHS = 1


dataset = InteriorDataset(TRAIN_DATA)
dataloader = DataLoader (
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else :
    device = torch.device("cpu")
print("Using device: ", device)


model = AutoEncoder().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    epoch_loss = 0
    for images in dataloader:
        images = images.to(device)
        output = model(images)
        loss = loss_fn(output, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")


torch.save(model.state_dict(), "checkpoints/autoencoder_epoch_1.pth")
print("Saved checkpoint to checkpoints/autoencoder_epoch_1.pth")