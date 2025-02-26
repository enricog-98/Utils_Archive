import os

if __name__ == "__main__":

    dataset_folder = "./datasets/VisDrone"

    for dataset_partition in os.listdir(dataset_folder):
        labels_folder = os.path.join(dataset_folder, dataset_partition, "labels")
        if not os.path.isdir(labels_folder):
            continue
        for label_file in os.listdir(labels_folder):
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
                        if line.split(" ")[0] == "1":
                            f.write("0" + line[1:] + "\n")
                        else:
                            f.write(line + "\n")
            except Exception as e:
                print(f"Error processing file {label_file_path}: {e}")