import os
import cv2
from tqdm import tqdm


def extract_classification_dataset(dataset_folder: str):
    """Extracts instances of each class from a detection dataset, transforming it into a classification dataset."""    
    for dataset_partition in ["train", "val", "test"]:
        images_path = os.path.join(dataset_folder, dataset_partition, "images")
        labels_path = os.path.join(dataset_folder, dataset_partition, "labels")
        output_path = os.path.join(dataset_folder, "extracted")
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Skipping {dataset_partition}, missing images or labels folder.")
            continue
        
        os.makedirs(output_path, exist_ok=True)
        
        label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
        for label_file in tqdm(label_files, desc=f"Processing {dataset_partition}"):
            if label_file.endswith(".txt"):
                image_file = label_file.replace(".txt", ".jpg")
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, label_file)
                
                if not os.path.exists(image_path):
                    print(f"Image {image_file} not found for label {label_file}, skipping.")
                    continue
                
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                
                with open(label_path, "r") as lf:
                    for idx, line in enumerate(lf.readlines()):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            print(f"Skipping invalid label line in {label_file}: {line}")
                            continue
                        
                        class_id = parts[0]
                        x_center, y_center, box_width, box_height = map(float, parts[1:5])
                        
                        x_center *= width
                        y_center *= height
                        box_width *= width
                        box_height *= height
                        
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)
                        
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                        object_crop = image[y1:y2, x1:x2]
                        
                        if object_crop.size == 0:
                            print(f"Skipping empty crop in {label_file}, line {idx+1}")
                            continue
                        
                        class_folder = os.path.join(output_path, class_id)
                        os.makedirs(class_folder, exist_ok=True)
                        
                        object_filename = f"{os.path.splitext(image_file)[0]}_{idx}.jpg"
                        object_filepath = os.path.join(class_folder, object_filename)
                        cv2.imwrite(object_filepath, object_crop)


if __name__ == "__main__":
    dataset_folder = "./datasets/..."

    for dataset_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset_name)
        if os.path.isdir(dataset_path):
            extract_classification_dataset(dataset_path)