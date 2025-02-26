import os
import cv2


def extract_classes_from_dataset(dataset_path):
    """Extracts instances of each class from a detection dataset, transforming it into a classification dataset."""
    subsets = ["train", "valid", "test"]
    
    for subset in subsets:
        images_path = os.path.join(dataset_path, subset, "images")
        labels_path = os.path.join(dataset_path, subset, "labels")
        output_path = os.path.join(dataset_path, "extracted")
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Skipping {subset}, missing images or labels folder.")
            continue
        
        os.makedirs(output_path, exist_ok=True)
        
        for label_file in os.listdir(labels_path):
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
    datasets_directory = "./datasets/1. Original detection datasets"

    for dataset in os.listdir(datasets_directory):
        dataset = os.path.join(datasets_directory, dataset)
        if os.path.isdir(dataset):
            extract_classes_from_dataset(dataset)