# ensure newly created processed data is good
from pathlib import Path 
from torchvision.io import read_image

# 1. Find the dataset folder
train_dir = Path("data/processed/interior_128/train")
print(train_dir.exists()) 
test_dir = Path("data/processed/interior_128/test")
print(test_dir.exists()) 

# 2. Count  images
train_data = list(train_dir.glob("**/*.jpg")) 
test_data = list(test_dir.glob("**/*.jpg")) 
print(f"Number of training images: {len(train_data)}")
print(f"Number of test images: {len(test_data)}")


# 3. inspect size and dtype and min max vals
image_path = train_data[0]
image_tensor = read_image(image_path)
print("Shape: ", image_tensor.shape)
print("Dtype: ", image_tensor.dtype)
print("Min: ", image_tensor.min().item())
print("Max: ", image_tensor.max().item())

# 4. ensure all images are consistent train set 

img_map = {}

for img in train_data :
    curr_img = read_image(img)
    if tuple(curr_img.shape) in img_map:
        img_map[tuple(curr_img.shape)] += 1
    else :
        img_map[tuple(curr_img.shape)] = 1
print(img_map)


# 5. ensure all images are consistent test set 
img_map1 = {}

for img in test_data :
    curr_img = read_image(img)
    if tuple(curr_img.shape) in img_map1:
        img_map1[tuple(curr_img.shape)] += 1
    else :
        img_map1[tuple(curr_img.shape)] = 1
print(img_map1)

# 6. all styles are consistent train 

for i in train_dir.iterdir():
    if i.is_dir():
        print(i.name)

# 7. all styles are consistent test
for i in test_dir.iterdir():
    if i.is_dir():
        print(i.name)