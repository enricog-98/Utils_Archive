import os
import random
import shutil
import torch
from ultralytics import YOLO
from tqdm import tqdm


def split_dataset(dataset_path, train_size: float):
    classes = os.listdir(dataset_path)
    for c in classes:
        images = os.listdir(os.path.join(dataset_path, c))
        random.shuffle(images)

        split = int(len(images) * train_size)
        train_images = images[:split]
        val_images = images[split:]
        
        os.makedirs(os.path.join(dataset_path, "train", c), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "test", c), exist_ok=True)
        
        for img in tqdm(train_images, desc=f"Copying train images for class {c}", leave=False):
            shutil.copy(os.path.join(dataset_path, c, img), os.path.join(dataset_path, "train", c, img))
        for img in tqdm(val_images, desc=f"Copying test images for class {c}", leave=False):
            shutil.copy(os.path.join(dataset_path, c, img), os.path.join(dataset_path, "test", c, img))
            

if __name__ == "__main__":
    dataset_path = "./datasets/PPE classification dataset"

    #split_dataset(dataset_path, train_size=0.8)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"{'#'*20}\nDevice used: {device}\n{'#'*20}")
        
    model = YOLO("yolo11n-cls.pt")  # n, s, m, l, x
    results = model.train(data = dataset_path,
                          #batch = 16,
                          classes = ["vest", "no-vest"],
                          epochs = 10,
                          plots = True,
                          device = device,
                          #val = False,
                          project = "runs/prova"
                          )