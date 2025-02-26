import torch
from ultralytics import YOLO


def train_yolo_classification(dataset_path: str, epochs: int = 1000):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"{'#'*20}\nDevice used: {device}\n{'#'*20}")
    
    model = YOLO("yolo11n-cls.pt")  # n, s, m, l, x
    results = model.train(data = dataset_path,
                          epochs = epochs,
                          plots = True,
                          device = device,
                          project = "runs/..."
                          )


if __name__ == "__main__":
    dataset_path = "./datasets/..."
    epochs = 1000

    train_yolo_classification(dataset_path, epochs)