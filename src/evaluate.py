"""
Comprehensive model evaluation and analysis for CIFAR-10 classification.

This module provides extensive evaluation capabilities including:
- Standard classification metrics (accuracy, F1-score, confusion matrix)
- Advanced visualization tools (GradCAM, t-SNE embeddings)
- Training curve plotting and analysis
- Model interpretability and feature visualization
- Comprehensive performance reporting
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict

from .enums import ModelType
from .data_loader import load_data
from .model import get_model

logger = logging.getLogger(__name__)

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates a classification model on a validation or test set.

    Args:
        model (nn.Module): Trained model to evaluate.
        val_loader (DataLoader): DataLoader for validation/test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.

    Returns:
        Tuple[float, float, float]: (average loss, accuracy %, weighted F1 score)
    """
    logger.info("Starting evaluation...")

    # --- Sanity checks ---
    if not isinstance(model, nn.Module):
        raise TypeError("Provided model is not a valid nn.Module.")

    if not hasattr(val_loader, "__iter__"):
        raise TypeError("val_loader must be iterable.")

    try:
        sample = next(iter(val_loader))
    except StopIteration:
        raise ValueError("val_loader is empty.")
    except Exception as e:
        raise RuntimeError(f"Could not fetch sample from val_loader: {e}")

    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError("Each batch from val_loader should be a tuple (images, labels).")

    images, labels = sample
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("val_loader should return torch.Tensor objects.")

    if images.dim() != 4:
        raise ValueError(f"Images should be 4D (B, C, H, W). Got shape: {images.shape}")

    if labels.dim() != 1:
        raise ValueError(f"Labels should be 1D. Got shape: {labels.shape}")

    if torch.isnan(images).any() or torch.isnan(labels).any():
        raise ValueError("val_loader contains NaN values.")

    logger.info("Evaluation sanity checks passed.")

    # --- Evaluation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if torch.isnan(outputs).any():
                raise ValueError("Model outputs contain NaNs during evaluation.")

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logger.info(f"Eval Results — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    return avg_loss, accuracy, f1

def plot_training_curves(
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    train_f1: Optional[List[float]] = None,
    val_f1: Optional[List[float]] = None
) -> None:
    """
    Plots training/validation loss, accuracy, and optional F1 scores over epochs.
    """
    logger.info("Plotting training curves...")
    epochs = range(1, len(train_loss) + 1)
    num_plots = 3 if train_f1 and val_f1 else 2
    plt.figure(figsize=(16, 5) if num_plots == 3 else (12, 5))

    plt.subplot(1, num_plots, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, num_plots, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.ylim(0, 100)
    plt.legend()

    if train_f1 and val_f1:
        plt.subplot(1, num_plots, 3)
        plt.plot(epochs, train_f1, label='Train F1')
        plt.plot(epochs, val_f1, label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Epochs')
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.show()
    logger.info("Training curves plotted.")
    

def final_evaluation(models_dir: str = "models", images_dir: str = "images") -> pd.DataFrame:
    """
    Evaluates all trained models on the test set.
    Saves confusion matrices and logs key metrics.

    Returns:
        pd.DataFrame: Evaluation summary.
    """
    logger.info("Starting final evaluation...")
    os.makedirs(images_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    logger.info(f"Using device: {device}")

    model_paths: Dict[ModelType, str] = {
        ModelType.RESNET: f"{models_dir}/resnet50-cifar10.pth",
        ModelType.EFFICIENTNET: f"{models_dir}/efficientnet-cifar10.pth",
        ModelType.VIT: f"{models_dir}/vit-cifar10.pth"
    }

    results: List[Dict[str, float]] = []

    for model_type, path in model_paths.items():
        logger.info(f"Evaluating model: {model_type.name} | Checkpoint: {path}")

        # --- Check if checkpoint exists ---
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            continue

        try:
            # Load model
            model = get_model(model_type=model_type, num_classes=10, pretrained=False).to(device)

            checkpoint = torch.load(path, map_location=device)
            if "model_state_dict" not in checkpoint:
                raise KeyError(f"'model_state_dict' missing in checkpoint: {path}")

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        except Exception as e:
            logger.exception(f"Failed to load model for {model_type.name}: {e}")
            continue

        # --- Load test data ---
        try:
            _, _, test_loader = load_data(model_type=model_type, batch_size=64)
            criterion = CrossEntropyLoss()

            # Sanity check
            sample = next(iter(test_loader))
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise ValueError("Test loader must return a tuple (images, labels)")

            if sample[0].dim() != 4:
                raise ValueError(f"Expected image shape (B, C, H, W), got {sample[0].shape}")
        except Exception as e:
            logger.exception(f"Failed to prepare test data for {model_type.name}: {e}")
            continue

        # --- Run predictions ---
        y_true, y_pred = [], []
        try:
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    if torch.isnan(outputs).any():
                        raise ValueError("Model outputs contain NaNs.")

                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
        except Exception as e:
            logger.exception(f"Prediction failed for {model_type.name}: {e}")
            continue

        # --- Confusion Matrix ---
        try:
            cm = confusion_matrix(y_true, y_pred)
            class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, "classes") else list(range(10))

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Confusion Matrix - {model_type.name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            filename = f"{images_dir}/confusion_matrix_{model_type.name.lower()}.png"
            plt.savefig(filename)
            plt.close()
            logger.info(f"Saved confusion matrix to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix for {model_type.name}: {e}")

        # --- Evaluation Metrics ---
        try:
            test_loss, test_acc, test_f1 = evaluate(
                model=model,
                val_loader=test_loader,
                criterion=criterion,
                device=device
            )

            logger.info(f"{model_type.name} — Test Loss: {test_loss:.4f}, "
                        f"Accuracy: {test_acc:.2f}%, F1 Score: {test_f1:.4f}")

            results.append({
                "Model": model_type.name,
                "Accuracy (%)": round(test_acc, 2),
                "F1 Score": round(test_f1, 4),
                "Loss": round(test_loss, 4)
            })

        except Exception as e:
            logger.exception(f"Evaluation failed for {model_type.name}: {e}")

    # --- Results Table ---
    if not results:
        logger.error("No models were successfully evaluated.")
        return pd.DataFrame(columns=["Model", "Accuracy (%)", "F1 Score", "Loss"])

    df_results = pd.DataFrame(results)
    df_results.sort_values("Accuracy (%)", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    logger.info("Final evaluation complete:\n" + df_results.to_markdown(index=False))

    return df_results

def plot_tsne(models_dir: str = "models", images_dir: str = "images") -> None:
    """
    Generates a t-SNE visualization of the ViT model’s learned embeddings for the CIFAR-10 test set.

    Steps:
    1. Load trained ViT model and its weights.
    2. Extract CLS token embeddings for each image in the test set.
    3. Apply t-SNE dimensionality reduction.
    4. Plot the resulting 2D embeddings, colored by class.

    Output:
        Saves t-SNE plot to: `images/vit_tsne_embeddings.png`
    """

    # -------------------------
    # 1. Load Trained ViT Model
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    vit_model = get_model(model_type=ModelType.VIT, num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(f"{models_dir}/vit-cifar10.pth", map_location=device)
    vit_model.load_state_dict(checkpoint["model_state_dict"])
    vit_model.eval()

    # Define a helper to extract intermediate features (CLS token)
    vit_model.forward_features_only = lambda x: vit_model.forward_features(x)

    # -------------------------
    # 2. Load Test Data
    # -------------------------
    _, _, test_loader = load_data(model_type=ModelType.VIT, batch_size=64)
    class_names: List[str] = test_loader.dataset.classes

    # -------------------------
    # 3. Extract Embeddings
    # -------------------------
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = vit_model.forward_features_only(imgs)

            if feats.ndim == 3:
                feats = feats[:, 0, :]  # Extract CLS token embedding: shape [B, C]

            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.concatenate(features, axis=0)  # Shape: [N, C]
    labels = np.array(labels)

    # -------------------------
    # 4. t-SNE Reduction
    # -------------------------
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    embeddings_2d = tsne.fit_transform(features)  # Shape: [N, 2]

    # -------------------------
    # 5. Plot Embeddings
    # -------------------------
    plt.figure(figsize=(10, 8))

    for i in range(10):
        idx = labels == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=class_names[i], alpha=0.6, s=20)

    plt.legend()
    plt.title("t-SNE of ViT Embeddings (CIFAR-10)")
    plt.tight_layout()

    output_path = f"{images_dir}/vit_tsne_embeddings.png"
    plt.savefig(output_path)
    print(f"Saved t-SNE plot to: {output_path}")
    plt.show()

def plot_grad_cam(models_dir: str = "models", images_dir: str = "images") -> None:
    """
    Generate and visualize Grad-CAM outputs for a ViT model trained on CIFAR-10.
    - Selects one representative image per class from the test set.
    - Computes Grad-CAM heatmaps using the last attention block.
    - Displays and saves a 2x5 image grid of the overlaid Grad-CAM outputs.

    Outputs:
        - Individual Grad-CAM visualizations saved to: `images/gradcam_vit/`
        - Grid of all Grad-CAM images saved as: `images/gradcam_vit/vit_gradcam_grid.png`
    """

    # -----------------------------
    # Setup model and device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # Load trained ViT model and checkpoint
    vit_model = get_model(model_type=ModelType.VIT, num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(f"{models_dir}/vit-cifar10.pth", map_location=device)
    vit_model.load_state_dict(checkpoint["model_state_dict"])
    vit_model.eval()

    # -----------------------------
    # Load test data and select 1 image per class
    # -----------------------------
    _, _, test_loader = load_data(model_type=ModelType.VIT, batch_size=64)
    class_names: List[str] = test_loader.dataset.classes

    seen_classes = set()
    selected_images = []
    selected_labels = []

    # Iterate through test set and collect one sample per class
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            cls = label.item()
            if cls not in seen_classes:
                selected_images.append(img)
                selected_labels.append(cls)
                seen_classes.add(cls)
            if len(seen_classes) == 10:
                break
        if len(seen_classes) == 10:
            break

    images_tensor = torch.stack(selected_images).to(device)
    labels_tensor = torch.tensor(selected_labels).to(device)

    # -----------------------------
    # Define reshape function for ViT Grad-CAM
    # -----------------------------
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape ViT output from (B, N+1, C) to (B, C, H, W) for Grad-CAM visualization.
        Removes CLS token and reshapes the sequence into 2D spatial format.
        """
        tensor = tensor[:, 1:, :]  # Remove CLS token
        B, N, C = tensor.shape
        H = W = int(N ** 0.5)
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)

    target_layers = [vit_model.blocks[-1].norm1]
    gradcam_dir = f"{images_dir}/gradcam_vit"
    os.makedirs(gradcam_dir, exist_ok=True)

    # -----------------------------
    # Generate and plot Grad-CAM
    # -----------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle("Grad-CAM Visualization per Class (ViT)", fontsize=20)

    with GradCAM(model=vit_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        for i in range(10):
            input_tensor = images_tensor[i].unsqueeze(0)
            label = labels_tensor[i].item()
            class_name = class_names[label]
            targets = [ClassifierOutputTarget(label)]

            # Generate Grad-CAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # shape: [H, W]

            # Convert image to RGB and overlay CAM
            rgb_img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Save individual CAM output
            out_path = f"{gradcam_dir}/vit_gradcam_class_{label}_{class_name}.png"
            plt.imsave(out_path, cam_image)
            print(f"Saved Grad-CAM: {out_path}")

            # Plot in grid
            row, col = divmod(i, 5)
            axes[row, col].imshow(cam_image)
            axes[row, col].set_title(class_name, fontsize=13)
            axes[row, col].axis("off")

    # Adjust layout and save the full grid
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for title
    grid_path = f"{gradcam_dir}/vit_gradcam_grid.png"
    plt.savefig(grid_path)
    print(f"Saved full Grad-CAM grid: {grid_path}")
    plt.show()

