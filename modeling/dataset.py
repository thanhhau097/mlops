import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.transform = transform

        self.image_paths, self.labels = self.get_image_paths_and_labels(root_folder)

    def get_image_paths_and_labels(self, root_folder):
        image_paths = []
        labels = []
        for label in os.listdir(root_folder):
            folder = os.path.join(root_folder, label)
            for image_name in os.listdir(folder):
                image_paths.append(os.path.join(folder, image_name))
                labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, int(label)

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert("L")
        image = image.resize((28, 28))
        image = np.array(image)
        return image
