import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from dataset import CustomImageDataset, split_dataset, save_validation_paths
import time
from tqdm import tqdm
import utils
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model to classify real vs. simulated images.')
    parser.add_argument('--dataset_path', type=str, default='../data', help='Path to your dataset folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu', help='Device to run the training on')
    parser.add_argument('--train_val_split', type=float, default=0.2, help='Ratio by which to split train and val data')
    return parser.parse_args()


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    log_file: "TextIO",
    device: torch.device
) -> None:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        criterion (nn.Module): Loss function.
        log_file ("TextIO"): File object to log training details.
        device (torch.device): Device to use for computation.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize progress bar
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, (inputs, labels, _) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate batch metrics
            batch_loss = loss.item()
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            wandb.log({
                'Batch Loss': batch_loss,
                'Batch Accuracy': batch_acc,
            })
            log_file.write(f"Epoch {epoch + 1}, Batch {batch_idx + 1}:\n")
            log_file.write(f"Batch Loss: {batch_loss:.4f}\n")
            log_file.write(f"Batch Accuracy: {batch_acc:.2f}%\n\n")
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss / total:.4f}',
                'Accuracy': f'{100 * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    log_file: "TextIO",
    device: torch.device
) -> tuple[float, float, float, float, float]:
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): Model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.
        log_file ("TextIO"): File object to log validation details.
        device (torch.device): Device to use for computation.

    Returns:
        tuple[float, float, float, float, float]: Validation loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    running_loss = 0.0
    misclassified_paths = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels, paths in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Calculate metrics
            val_loss = running_loss / len(val_loader.dataset)
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions)

            wandb.log({
                'Validation Loss': val_loss,
                'Validation Accuracy': accuracy,
                'Validation Precision': precision,
                'Validation Recall': recall,
                'Validation F1 Score': f1
            })
            log_file.write(f"Epoch {epoch + 1}:\n")
            log_file.write(f"Validation Loss: {val_loss:.4f}\n")
            log_file.write(f"Validation Accuracy: {accuracy:.4f}\n")
            log_file.write(f"Validation Precision: {precision:.4f}\n")
            log_file.write(f"Validation Recall: {recall:.4f}\n")
            log_file.write(f"Validation F1 Score: {f1:.4f}\n\n")
            # Identify misclassified images
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_paths.append(paths[i])

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save misclassified image paths
    misclassified_paths.sort()
    file_path = os.path.join(os.path.dirname(log_file.name), f'failed_val_epoch{epoch+1}.txt')
    with open(file_path, 'w') as f:
        for path in misclassified_paths:
            f.write(f"{path}\n")

    return val_loss, accuracy, precision, recall, f1

def main(args: argparse.Namespace, class_weights: torch.Tensor) -> None:
    """
    Main function to train and validate the model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        class_weights (torch.Tensor): Class weights for handling imbalance.
    """
    # Hyperparameter
    DATASET_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    wandb.init(project='real_vs_simulated_images')

    # Load dataset
    dataset = CustomImageDataset(DATASET_PATH)
    train_dataset, val_dataset = split_dataset(dataset, val_split=args.train_val_split)
    save_validation_paths(val_dataset, dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained ResNet-18 model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    model = model.to(DEVICE)

    # Define loss function with class weights to handle class imbalance
    class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    log_foldername = datetime.now().strftime('../logs/train_%Y%m%d_%H%M%S')
    model_save_path = datetime.now().strftime("best_model_%Y%m%d_%H%M%S.pth")
    os.makedirs(log_foldername, exist_ok=True)
    log_filename = os.path.join(log_foldername, 'training.txt')
    best_val_acc = -1
    early_stopping = utils.EarlyStopping(patience=5, verbose=True, path=model_save_path)
    with open(log_filename, 'a') as log_file:
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_one_epoch(epoch, model, train_loader, optimizer, criterion, log_file, DEVICE)
            _, val_acc, _, _, _ = validate(model, val_loader, criterion, epoch, log_file, DEVICE)
            end_time = time.time()

            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds")
            # If early stopping flag is set, break the loop
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Save the model if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved with Validation Accuracy: {best_val_acc:.2f}%")

    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    num_synthetic = 9471
    num_real = 4454
    total = num_synthetic + num_real
    weight_synthetic = total / (2 * num_synthetic)
    weight_real = total / (2 * num_real)
    class_weights = torch.tensor([weight_synthetic, weight_real])
    main(args, class_weights)