"""
Train Cats vs Dogs classifier with MLflow experiment tracking.

Usage:
    python scripts/train.py --data_dir data/raw --model_type simple_cnn --epochs 15
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from catsml.data import load_dataset, CLASS_NAMES
from catsml.model import get_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_loss_curves(train_losses, val_losses, train_accs, val_accs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw", help="Path to dataset root")
    parser.add_argument("--model_type", default="simple_cnn", choices=["simple_cnn", "transfer_resnet18"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="models", help="Where to save trained model")
    parser.add_argument("--mlflow_uri", default="mlruns", help="MLflow tracking URI")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_loader, val_loader, test_loader = load_dataset(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Dataset sizes - train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    model = get_model(args.model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("cats-vs-dogs")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device,
        })

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": vl_loss, "val_acc": vl_acc,
            }, step=epoch)
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        # Test evaluation
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

        # Save artifacts
        cm_path = "artifacts/confusion_matrix.png"
        save_confusion_matrix(test_labels, test_preds, cm_path)
        mlflow.log_artifact(cm_path)

        curves_path = "artifacts/loss_curves.png"
        save_loss_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
        mlflow.log_artifact(curves_path)

        # Save model
        model_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Save model metadata
        meta = {"model_type": args.model_type, "test_acc": test_acc, "class_names": CLASS_NAMES}
        meta_path = os.path.join(args.output_dir, "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        mlflow.log_artifact(meta_path)

        print(f"\nModel saved to {model_path}")
        print(f"MLflow run completed.")


if __name__ == "__main__":
    main()
