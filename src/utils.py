from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple
from PIL import ImageOps
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0, path: str = 'best_model.pth'):
        """
        EarlyStopping class to stop training if validation accuracy does not improve after a given patience.

        Args:
            patience (int): Number of epochs to wait after last improvement before stopping training.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change in validation accuracy to qualify as an improvement.
            path (str): File path to save the best model.
        """
        self.patience: int = patience  # Number of epochs to wait before stopping
        self.verbose: bool = verbose  # Verbosity flag
        self.delta: float = delta  # Minimum improvement threshold
        self.path: str = path  # Path to save the best model
        self.counter: int = 0  # Counter for epochs without improvement
        self.best_score: Optional[float] = None  # Best validation score
        self.early_stop: bool = False  # Flag to indicate early stopping
        self.best_acc: float = 0  # Best validation accuracy

    def __call__(self, val_acc: float, model: torch.nn.Module) -> None:
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc: float, model: torch.nn.Module) -> None:
        """Saves model when validation accuracy improves."""
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_acc = val_acc

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess a single image by resizing while maintaining aspect ratio, padding to the target size, and normalizing.

    Steps:
        1. Open the image and ensure it is in RGB format.
        2. Resize the image to fit within the target size, maintaining the aspect ratio.
        3. Pad the resized image to exactly match the target size.
        4. Convert the image to a tensor and normalize it.

    Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int]): Desired output size (height, width).

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (C, H, W).
    """
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

    # Step 1: Resize while maintaining aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:  # Landscape
        new_width = target_size[1]
        new_height = int(new_width / aspect_ratio)
    else:  # Portrait or square
        new_height = target_size[0]
        new_width = int(new_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Step 2: Pad to target size
    delta_w = target_size[1] - new_width
    delta_h = target_size[0] - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, fill=(0, 0, 0))

    # Step 3: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image)


