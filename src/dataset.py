import os
import glob
from torch.utils.data import Dataset, random_split
import utils

class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading synthetic and real images for classification.
    """
    def __init__(self, root_dir):
        self.root_dir: str = root_dir
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        # Load synthetic images
        synthetic_dir = os.path.join(root_dir, 'synthetic')
        synthetic_images = glob.glob(os.path.join(synthetic_dir, '*.png'))
        self.image_paths.extend(synthetic_images)
        self.labels.extend([0] * len(synthetic_images))  # Label 0 for synthetic
        print(f"Total synthetic images: {len(synthetic_images)}")

        # Load real images from subfolders
        real_dir = os.path.join(root_dir, 'real')
        real_image_count = 0
        for subfolder in os.listdir(real_dir):
            subfolder_path = os.path.join(real_dir, subfolder, 'images')
            if os.path.isdir(subfolder_path):
                real_images = glob.glob(os.path.join(subfolder_path, '*.jpg'))
                self.image_paths.extend(real_images)
                self.labels.extend([1] * len(real_images))  # Label 1 for real
                real_image_count += len(real_images)
        print(f"Total real images: {real_image_count}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = utils.preprocess_image(img_path)

        return image, label, img_path


def split_dataset(dataset: Dataset, val_split: float = 0.2) -> tuple[Dataset, Dataset]:
    """
    Split the dataset into training and validation datasets.

    Args:
        dataset (Dataset): The complete dataset to split.
        val_split (float): Fraction of data to use for validation.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def save_validation_paths(val_dataset: Dataset, original_dataset: CustomImageDataset, filename: str = '../data/val.txt') -> None:
    """
    Save the file paths of validation images to a file.

    Args:
        val_dataset (Dataset): Validation dataset containing indices of validation samples.
        original_dataset (CustomImageDataset): Original dataset for mapping indices to paths.
        filename (str): Path to the file where validation paths will be saved.
    """
    with open(filename, 'w') as f:
        for idx in val_dataset.indices:
            img_path = original_dataset.image_paths[idx]
            f.write(f"{img_path}\n")


if __name__ == "__main__":
    # Example usage
    root_dir = '../data'
    dataset = CustomImageDataset(root_dir)
    train_dataset, val_dataset = split_dataset(dataset)
    save_validation_paths(val_dataset, dataset)

