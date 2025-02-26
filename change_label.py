import os
from tqdm import tqdm


def change_label(dataset_folder: str, source_label: str, target_label: str):
    """Change the label of all objects with the source label to the target label in the dataset."""
    for dataset_partition in ["train", "val", "test"]:
        labels_folder = os.path.join(dataset_folder, dataset_partition, "labels")
        label_files = os.listdir(labels_folder)
        for label_file in tqdm(label_files, desc=f"Processing {dataset_partition} partition"):
            label_file_path = os.path.join(labels_folder, label_file)
            if not os.path.isfile(label_file_path):
                continue
            try:
                with open(label_file_path, "r") as f:
                    lines = f.readlines()
                with open(label_file_path, "w") as f:
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.split(" ")[0] == source_label:
                            f.write(target_label + line[1:] + "\n")
                        else:
                            f.write(line + "\n")
            except Exception as e:
                print(f"Error processing file {label_file_path}: {e}")


if __name__ == "__main__":
    dataset_folder = "./datasets/..."
    source_label = "0"
    target_label = "1"
    
    change_label(dataset_folder, source_label, target_label)