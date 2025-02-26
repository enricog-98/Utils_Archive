import os
import requests
from tqdm import tqdm


def generate_faces(folder: str, num_images: int):
    """Downloads AI-generated faces and saves them to a local folder."""
    os.makedirs(folder, exist_ok=True)
    url = "https://thispersondoesnotexist.com/"
    
    for i in tqdm(range(num_images), desc="Downloading faces"):
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            file_path = os.path.join(folder, f"face_{i}.jpg")
            with open(file_path, "wb") as file:
                file.write(response.content)


if __name__ == "__main__":
    download_folder = "./datasets/2. Extracted classification datasets/Synthetic Faces"
    num_images = 10 # Number of images you want to generate
    generate_faces(download_folder, num_images)