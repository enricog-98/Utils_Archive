import os
import random
import shutil
from tqdm import tqdm


def split_classification_dataset(dataset_path: str, train_size: float):
    classes = os.listdir(dataset_path)
    for class_name in classes:
        images = os.listdir(os.path.join(dataset_path, class_name))
        random.shuffle(images)

        split = int(len(images) * train_size)
        train_images = images[:split]
        val_images = images[split:]
        
        os.makedirs(os.path.join(dataset_path, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "val", class_name), exist_ok=True)
        
        for img in tqdm(train_images, desc=f"Copying train images for class {class_name}", leave=False):
            shutil.copy(os.path.join(dataset_path, class_name, img), os.path.join(dataset_path, "train", class_name, img))
        for img in tqdm(val_images, desc=f"Copying val images for class {class_name}", leave=False):
            shutil.copy(os.path.join(dataset_path, class_name, img), os.path.join(dataset_path, "val", class_name, img))


if __name__ == "__main__":
    dataset_path = "./datasets/..."

    split_classification_dataset(dataset_path, train_size=0.9)