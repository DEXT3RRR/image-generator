import torch
from pathlib import Path
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image
from torchvision.utils import save_image

train_dir = Path("data/raw/interior-design-styles/dataset_train/dataset_train")
test_dir = Path("data/raw/interior-design-styles/dataset_test/dataset_test")

train_data = list(train_dir.glob("**/*.jpg")) 
test_data = list(test_dir.glob("**/*.jpg")) 

# resize all images

resize = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToDtype(torch.float32, scale=True),
 ])

folder_dir = Path("data/processed/interior_128/train")
for img in train_data :
    relative_path =img.relative_to(train_dir)
    curr_img = read_image(str(img))
    processed_img = resize(curr_img)
    
    save_path = folder_dir / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(processed_img, str(save_path))

    

folder_dir = Path("data/processed/interior_128/test")
for img in test_data :
    relative_path =img.relative_to(test_dir)
    curr_img = read_image(str(img))
    processed_img = resize(curr_img)
    
    save_path = folder_dir / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(processed_img, str(save_path))

  

 



