from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms


class InteriorDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.paths = sorted(list(self.root_dir.glob("**/*.jpg")))
        if transform is None :
            self.transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
        ])
        else :
            self.transform = transform


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = read_image(str(img_path))
        img = self.transform(img)

        return img
    


# if __name__ == "__main__":
#     train_dataset = InteriorDataset("data/processed/interior_128/train")

#     print("Dataset length:", len(train_dataset))

#     sample_img = train_dataset[0]

#     print("Sample shape:", sample_img.shape)
#     print("Sample dtype:", sample_img.dtype)
#     print("Sample min:", sample_img.min().item())
#     print("Sample max:", sample_img.max().item())