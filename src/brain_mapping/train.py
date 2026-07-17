"""Training loop: mixed precision, gradient accumulation, early stopping.

Extracted from the original script (Trainer). Logic is unchanged.
"""

import json
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .models import CombinedLoss
from .utils import logger

# ======================== Training ========================


class Trainer:
    """Enhanced trainer with mixed precision, early stopping, and logging"""

    def __init__(self, model: nn.Module, config: TrainingConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.scaler = GradScaler() if config.use_amp else None
        self.criterion = CombinedLoss()

        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {"ce": 0, "dice": 0, "focal": 0}

        self.optimizer.zero_grad()

        for i, (imgs, masks, _) in enumerate(tqdm(dataloader, desc="Training")):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            # Mixed precision training
            if self.scaler:
                with autocast():
                    preds = self.model(imgs)
                    loss, components = self.criterion(preds, masks)
                    loss = loss / self.config.accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.config.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                preds = self.model(imgs)
                loss, components = self.criterion(preds, masks)
                loss = loss / self.config.accumulation_steps
                loss.backward()

                if (i + 1) % self.config.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += components["total"]
            for k in loss_components:
                loss_components[k] += components[k]

        n = len(dataloader)
        return {"loss": total_loss / n, **{k: v / n for k, v in loss_components.items()}}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        total_loss = 0
        dice_scores = []

        with torch.no_grad():
            for imgs, masks, _ in tqdm(dataloader, desc="Validation"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                preds = self.model(imgs)
                loss, _ = self.criterion(preds, masks)
                total_loss += loss.item()

                # Compute Dice score
                probs = F.softmax(preds, dim=1)[:, 1]
                pred_mask = (probs > 0.5).float()
                dice = self.compute_dice(pred_mask, masks.float())
                dice_scores.append(dice)

        return {"loss": total_loss / len(dataloader), "dice": np.mean(dice_scores)}

    @staticmethod
    def compute_dice(pred, target, smooth=1e-5):
        """Compute Dice coefficient"""
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2.0 * inter + smooth) / (union + smooth)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str = "best_model.pth"):
        """Full training loop with early stopping"""
        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - Train Loss: {train_metrics['loss']:.4f}"
            )

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Save history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["metrics"].append(val_metrics)

            # Early stopping
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": self.best_loss,
                        "config": asdict(self.config),
                    },
                    save_path,
                )
                logger.info(f"Model saved with loss: {self.best_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save training history
        with open(save_path.replace(".pth", "_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        return self.model
