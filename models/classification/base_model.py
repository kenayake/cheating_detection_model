import os
from torch import Tensor
import torch.nn as nn
import time
from torch.optim import Optimizer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models
from wakepy import keep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.tensorboard import SummaryWriter

from roboflow import Roboflow

from models.classification.dataset_loader import DatasetLoader, get_class_weights
from models.classification.focal_loss import FocalLoss
from scipy.ndimage import gaussian_filter1d


class BaseModel:
    def __init__(
        self,
        model: nn.Module,
        weights,
        data_dir,
        save_dir,
        optimizer: Optimizer,
        criterion,
        batch_size=16,
    ):
        self.model = model
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.criterion = criterion
        self.classes = {
            0: "hands_not_visible",
            1: "look_around",
            2: "look_behind",
            3: "not_cheating",
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = DatasetLoader(
            transforms=weights.transforms(), batch_size=batch_size
        )
        self.stopped_early = False
        self.writer = SummaryWriter(log_dir=save_dir)

    def plot_confusion_matrix(self, cm, class_names, filename):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({"best" if self.stopped_early else "last"})")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()

    def plot_metrics(self, metrics_dict, filename):
        plt.figure(figsize=(6, 4))
        names = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        bars = plt.bar(names, values, color=["skyblue", "orange", "green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Test Metrics ({"best" if self.stopped_early else "last"})")
        # Annotate each bar with its value
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{filename}")
        plt.close()

    def plot_curves(
        self, train_losses, val_losses, val_accuracies, val_precisions, val_recalls
    ):

        # Plot and save training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/loss_curve.png")
        plt.close()

        # Plot and save validation accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/val_accuracy_curve.png")
        plt.close()

        # Plot and save validation precision curve
        plt.figure(figsize=(10, 6))
        plt.plot(val_precisions, label="Validation Precision", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.title("Validation Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/val_precision_curve.png")
        plt.close()

        # Plot and save validation recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(val_recalls, label="Validation Recall", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.title("Validation Recall")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/val_recall_curve.png")
        plt.close()

    def write_to_tensorboard(
        self,
        train_losses,
        val_losses,
        val_accuracies,
        val_precisions,
        val_recalls,
        epoch,
    ):
        self.writer.add_scalars(
            "Loss", {"train": train_losses, "val": val_losses}, epoch
        )
        self.writer.add_scalar("Accuracy/val", val_accuracies, epoch)
        self.writer.add_scalar("Precision/val", val_precisions, epoch)
        self.writer.add_scalar("Recall/val", val_recalls, epoch)

    def load_pretrained_model(self):
        num_classes = len(self.classes)
        last_layer_name, last_layer_val = list(self.model.named_modules())[-1]
        self.model.set_submodule(
            last_layer_name, nn.Linear(last_layer_val.in_features, num_classes)
        )
        self.model = self.model.to(self.device)

    def evaluate(self, model, loader):
        model.eval()
        all_labels = []
        all_preds = []
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        return avg_loss, accuracy, precision, recall, cm, all_labels, all_preds

    def train(self, num_epochs):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        train_loader, val_loader, test_loader = self.dataset.load_data(self.data_dir)
        # Initialize lists to store metrics
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []

        # Early stopping parameters
        early_stop_patience = 30
        best_val_loss = float("inf")
        early_stop_counter = 0
        best_model = None
        best_epoch = 0

        self.load_pretrained_model()

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * labels.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            val_loss, val_acc, val_prec, val_rec, val_cm, _, _ = self.evaluate(
                self.model, val_loader
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_precisions.append(val_prec)
            val_recalls.append(val_rec)

            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}"
            )
            print(f"Confusion Matrix:\n{val_cm}")

            self.write_to_tensorboard(
                train_loss, val_loss, val_acc, val_prec, val_rec, epoch
            )

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_epoch = epoch + 1
            else:
                early_stop_counter += 1
                print(
                    f"Early stopping counter: {early_stop_counter}/{early_stop_patience}"
                )

            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered. Stopping training.")
                self.stopped_early = True
                best_model = self.model
                torch.save(
                    best_model.state_dict(), f"{self.save_dir}/best@{best_epoch}.pth"
                )
                break

        test_loss, test_acc, test_prec, test_rec, test_cm, test_labels, test_preds = (
            self.evaluate(best_model or self.model, test_loader)
        )
        print(
            f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Prec: {test_prec:.4f} | Test Rec: {test_rec:.4f}"
        )
        print(f"Test Confusion Matrix:\n{test_cm}")

        # Output metrics as images
        self.plot_confusion_matrix(
            test_cm, [*self.classes.values()], "test_confusion_matrix.png"
        )
        metrics_dict = {
            "Accuracy": test_acc,
            "Precision": test_prec,
            "Recall": test_rec,
        }
        self.plot_metrics(metrics_dict, "test_metrics.png")
        print(
            "Test metrics and confusion matrix images saved as 'test_metrics.png' and 'test_confusion_matrix.png'"
        )
        self.plot_curves(
            train_losses, val_losses, val_accuracies, val_precisions, val_recalls
        )

        torch.save(self.model.state_dict(), f"{self.save_dir}/last.pth")

        self.writer.close()

    def load_checkpoint(self, checkpoint_path):
        self.load_pretrained_model()
        # Load the trained checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict_image(self, image):

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            print(probabilities)
            confidence, predicted_idx = torch.max(probabilities, 1)

        return self.classes[predicted_idx.item()], confidence.item()


def initialize_roboflow(dt_version):
    rf = Roboflow(api_key="m2wBNOl1tzsXZBfl8Lwv")
    project = rf.workspace("skripsi-gfit0").project("cheating-classification")
    version = project.version(dt_version)
    dataset = version.download(
        "folder",
        location=f"dataset/classification/cheating-classification-v{dt_version}",
    )
    return dataset


def multi_train(
    model_group,
    dt_version,
    models_dict,
    base_weights,
    optims,
    epochs,
    batch_size=32,
    criterion=None,
):
    with keep.presenting():
        dataset = initialize_roboflow(dt_version)
        data_dir = dataset.location
        class_weights = get_class_weights(data_dir)
        criterion = criterion or FocalLoss(
            alpha=class_weights, task_type="multi-class", num_classes=len(class_weights)
        )
        for index, (model_name, model_object) in enumerate(models_dict.items()):
            weights = base_weights[index]
            base_model = model_object(weights=weights)
            print(f"Training {model_name} with weights {weights}...")

            # Debugging: Check if parameters exist
            params = list(base_model.parameters())
            if not params:
                print(f"Error: {model_name} has no parameters. Skipping...")
                continue

            if not any(param.requires_grad for param in params):
                print(f"Error: {model_name} has no trainable parameters. Skipping...")
                continue

            optimizers = optims(params)
            for optim_name, optimizer in optimizers.items():
                savedir = f"saved_checkpoints/{model_group}/{model_name}_dtv{dt_version}_{optim_name}_batch{batch_size}"

                trainer = BaseModel(
                    base_model,
                    weights,
                    data_dir,
                    savedir,
                    optimizer,
                    criterion,
                    batch_size=batch_size,
                )
                trainer.train(epochs)


if __name__ == "__main__":
    model = BaseModel(
        model=models.resnet152(weights=models.ResNet152_Weights.DEFAULT),
        classes={
            0: "giving_code",
            1: "giving_object",
            2: "looking_friend",
            3: "looking_phone",
            4: "not_cheating",
        },
    )

    model.load_pretrained_model()
