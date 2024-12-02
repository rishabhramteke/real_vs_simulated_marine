import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from dataset import CustomImageDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet-18 model on a test dataset.")
    parser.add_argument('--dataset_path', type=str, default='../data/test', help='Path to the test dataset folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--model_path', type=str, default='../models/best_model.pth', help='Path to the saved model file')
    return parser.parse_args()

def test(args: argparse.Namespace):
    """
    Evaluate a trained ResNet-18 model on the test dataset.

    Args:
        args (argparse.Namespace): Command-line arguments with dataset path, batch size, and model path.
    """
    # Device configuration
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # Load test dataset
    test_dataset = CustomImageDataset(args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the trained model
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # Binary classification
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Evaluation
    all_labels = []
    all_predictions = []
    misclassified_paths = []

    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified_paths.append(paths[i])

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)

    # Save misclassified image paths
    misclassified_paths.sort()
    file_path = os.path.join('../logs', f'failed_test_model_{os.path.basename(args.model_path).replace(".pth","")}.txt')
    with open(file_path, 'w') as f:
        for path in misclassified_paths:
            f.write(f"{path}\n")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test(args)
