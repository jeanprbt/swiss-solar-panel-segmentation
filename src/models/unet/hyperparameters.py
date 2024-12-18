import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

import torch.nn as nn
import torch.optim as optim
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from model_functions import evaluate
from models.unet.architecture import UNet


# ------------------------------------- Segmentation Threshold -----------------------------------------
def optimize_threshold(
    model: nn.Module, test_loader: DataLoader, n_trials: int = 10, device: torch.device = torch.device("cpu")
) -> float:
    """
    Optimize the threshold of a semantic segmentation model using Optuna.

    Args:
        model (nn.Module): A PyTorch semantic segmentation model accepting images as input and outputting masks.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.
        n_trials (int): Number of trials to run.

    Returns:
        float: Best threshold found by Optuna.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: threshold_objective(trial, model, test_loader, device), n_trials=n_trials)
    fig = optuna.visualization.plot_optimization_history(study, target_name="F1-Score")
    fig.show()
    return study.best_params["threshold"]


def threshold_objective(
    trial: optuna.trial.Trial, model: nn.Module, test_loader: DataLoader, device: torch.device = torch.device("cpu")
) -> float | float:
    """
    Objective function for Optuna to optimize the threshold of a semantic segmentation model.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.
        model (nn.Module): A PyTorch semantic segmentation model accepting images as input and outputting masks.
        test_loader (torch.utils.data.dataloader.DataLoader): DataLoader for test data.

    Returns:
        float: F1-Score.
        float: Intersection over Union (IoU) score.
    """
    threshold = trial.suggest_float("threshold", 0.1, 0.9)
    _, f1, _ = evaluate(model=model, test_loader=test_loader, threshold=threshold, device=device)
    return f1


# --------------------------------------- UNet hyperparameters -----------------------------------------
def optimize_unet_hyperparameters(
    train_loader: DataLoader, val_loader: DataLoader, n_trials: int = 10, device: torch.device = torch.device("cpu")
) -> dict:
    """
    Optimize hyperparameters of a U-Net model using Optuna, namely learning rate, number of
    epochs, number of layers and kernel size.

    Args:
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.
        n_trials (int): Number of trials to run.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: unet_objective(trial, train_loader, val_loader, device), n_trials=n_trials)
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    return study.best_params


def unet_objective(
    trial: optuna.trial.Trial, train_loader: DataLoader, val_loader: DataLoader, device: torch.device = torch.device("cpu")
) -> float:
    """
    Objective function for Optuna to optimize hyperparameters of a U-Net model, i.e learning rate,
    number of epoches, number of layers and kernel size.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.
        train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for validation data.

    Returns:
        float: Validation loss.
    """
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 20, 40, step=5)
    
    nb_layers = trial.suggest_int("encoder_layers", 2, 3)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    
    # Define a custom U-Net model
    model = UNet(nb_layers, kernel_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping params
    patience = 3
    best_val_loss = float("inf")
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for image, label, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for image, label, _ in val_loader:
                image, label = image.to(device), label.to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        # Early stopping
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    return val_loss