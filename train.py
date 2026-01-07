"""
Training script for Atlas-free Brain Network Transformer (BrainNet).
Supports command-line arguments for flexible training configuration.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from dataset import MRIDataset
from brainnet import BrainNet, ModelConfig


def setup_logging(log_file: str = 'train_log.txt'):

    logger_obj = logging.getLogger(__name__)
    
    for handler in logger_obj.handlers[:]:
        logger_obj.removeHandler(handler)
    
    logger_obj.setLevel(logging.INFO)
    
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger_obj.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger_obj.addHandler(file_handler)
    
    return logger_obj


logger = logging.getLogger(__name__)


class TrainingConfig:
    
    def __init__(self, args=None):
        if args is None:
            args = {}
        
        timestamp = int(time.time())
        base_dir = './training_runs'
        os.makedirs(base_dir, exist_ok=True)
        self.training_dir = os.path.join(base_dir, f'run_{timestamp}')
        os.makedirs(self.training_dir, exist_ok=True)
        
        self.data_dir = args.get('data_dir', './toy_data/')
        self.train_csv = args.get('train_csv', './data_split/train_df.csv')
        self.val_csv = args.get('val_csv', './data_split/val_df.csv')
        
        self.batch_size = args.get('batch_size', 16)
        self.num_epochs = args.get('num_epochs', 50)
        self.learning_rate = args.get('learning_rate', 1e-4)
        self.weight_decay = args.get('weight_decay', 1e-3)
        self.early_stopping_patience = args.get('early_stopping_patience', 10)
        self.lr_scheduler_patience = args.get('lr_scheduler_patience', 5)
        self.num_workers = args.get('num_workers', 4)
        
        self.checkpoint_path = os.path.join(self.training_dir, 'best_model.pth')
        self.log_file = os.path.join(self.training_dir, 'train_log.txt')
        self.curves_file = os.path.join(self.training_dir, 'training_curves.jpg')
        self.device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def to_dict(self) -> Dict:
        return self.__dict__


class Trainer:
    
    def __init__(self, config: TrainingConfig):

        self.config = config
        self.device = config.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model, loss function, and optimizer
        self.model = self._setup_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        
    def _setup_model(self):

        model_config = ModelConfig()
        model = BrainNet(model_config)
        model = model.to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _setup_optimizer(self):

        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.lr_scheduler_patience,
            min_lr=1e-6
        )
        return scheduler
    
    def _train_epoch(self, train_loader: DataLoader):

        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for f_mat, c_mat, labels in pbar:
            
            f_mat = f_mat.to(self.device)  # [B, 400, 1632]
            c_mat = c_mat.to(self.device)  # [B, 45, 54, 45]
            labels = labels.squeeze().to(self.device)  # [B]
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(f_mat, c_mat)  # [B, 2]
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        avg_loss = train_loss / len(train_loader)
        accuracy = 100 * train_correct / train_total
        return avg_loss, accuracy
    
    def _validate(self, val_loader: DataLoader):

        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for f_mat, c_mat, labels in pbar:
                # Move data to device
                f_mat = f_mat.to(self.device)
                c_mat = c_mat.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                logits = self.model(f_mat, c_mat)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * val_correct / val_total
        return avg_loss, accuracy
    
    def _save_checkpoint(self, epoch: int, train_acc: float, val_acc: float):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
        }

        torch.save(checkpoint, self.config.checkpoint_path)
        logger.info(f"✓ Saved checkpoint to {self.config.checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            
            # Store metrics for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {lr:.2e}"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, train_acc, val_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"\n Early stopping!")
                    logger.info(
                        f"Validation loss hasn't improved for "
                        f"{self.config.early_stopping_patience} epochs"
                    )
                    break
        
        # Training complete
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        logger.info("=" * 70)
        
        # Save training curves
        self._save_training_curves()
    
    def _save_training_curves(self):
        """
        Plot and save training/validation loss and accuracy curves.
        """
        output_file = self.config.curves_file
        if not self.train_losses or not self.train_accs:
            logger.warning("No training metrics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy curves
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, format='jpg', bbox_inches='tight')
        logger.info(f"✓ Saved training curves to {output_file}")
        plt.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Atlas-free Brain Network Transformer (BrainNet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./toy_data/',
        help='Path to directory containing .mat files (default: %(default)s)'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='./data_split/train_df.csv',
        help='Path to training CSV file (default: %(default)s)'
    )
    parser.add_argument(
        '--val_csv',
        type=str,
        default='./data_split/val_df.csv',
        help='Path to validation CSV file (default: %(default)s)'
    )

    
    # Training hyperparameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training (default: %(default)s)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: %(default)s)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: %(default)s)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-3,
        help='Weight decay for regularization (default: %(default)s)'
    )
    
    # Early stopping and scheduling
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Early stopping patience in epochs (default: %(default)s)'
    )
    parser.add_argument(
        '--lr_scheduler_patience',
        type=int,
        default=5,
        help='Learning rate scheduler patience in epochs (default: %(default)s)'
    )
    
    # Misc arguments
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading (default: %(default)s)'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='best_model.pth',
        help='Path to save best model checkpoint (default: %(default)s)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    return parser.parse_args()


def main():

    args = parse_arguments()
    
    args_dict = vars(args)
    config = TrainingConfig(args_dict)
    
    # Setup logging to training directory
    global logger
    logger = setup_logging(config.log_file)
    
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Loading datasets...")
    try:
        train_df = pd.read_csv(config.train_csv)
        val_df = pd.read_csv(config.val_csv)
        
        train_dataset = MRIDataset(config.data_dir, train_df)
        val_dataset = MRIDataset(config.data_dir, val_df)
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()