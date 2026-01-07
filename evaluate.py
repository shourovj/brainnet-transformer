"""
Evaluation script for Atlas-free Brain Network Transformer (BrainNet).
Supports command-line arguments for flexible testing configuration.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import MRIDataset
from brainnet import BrainNet, ModelConfig


def setup_logging(log_file: str = 'eval_log.txt'):
    """
    Setup logging to both console and file.
    
    Args:
        log_file: Path to log file
    """
    logger_obj = logging.getLogger(__name__)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger_obj.handlers[:]:
        logger_obj.removeHandler(handler)
    
    logger_obj.setLevel(logging.INFO)
    
    # Log format
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger_obj.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger_obj.addHandler(file_handler)
    
    return logger_obj


logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration class for evaluation hyperparameters."""
    
    def __init__(self, args=None):
        """Initialize evaluation configuration from arguments or defaults."""
        if args is None:
            args = {}
        
        # Create timestamped evaluation directory
        timestamp = int(time.time())
        base_dir = './evaluation_runs'
        os.makedirs(base_dir, exist_ok=True)
        self.evaluation_dir = os.path.join(base_dir, f'run_{timestamp}')
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        self.data_dir = args.get('data_dir', './toy_data/')
        self.test_csv = args.get('test_csv', './data_split/test_df.csv')
        self.checkpoint_path = args.get('checkpoint_path', 'best_model.pth')
        
        self.batch_size = args.get('batch_size', 16)
        self.num_workers = args.get('num_workers', 4)
        
        self.device = args.get('device', None)
        
        # Save all outputs to evaluation directory
        self.log_file = os.path.join(self.evaluation_dir, 'eval_log.txt')
        self.confusion_matrix_file = os.path.join(self.evaluation_dir, 'confusion_matrix.jpg')
        self.metrics_file = os.path.join(self.evaluation_dir, 'evaluation_metrics.jpg')
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__


class Evaluator:
    """Main evaluator class for model testing and evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator.
        
        Args:
            config: EvaluationConfig instance with evaluation hyperparameters
        """
        self.config = config
        self.device = config.device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._setup_model()
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_model(self) -> BrainNet:
        """Setup and initialize model."""
        model_config = ModelConfig()
        model = BrainNet(model_config)
        model = model.to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _load_checkpoint(self) -> Dict:
        """
        Load model checkpoint.
        
        Returns:
            Dictionary containing checkpoint information
        """
        try:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Loaded checkpoint from {self.config.checkpoint_path}")
            return checkpoint
        except FileNotFoundError:
            logger.error(f"Checkpoint not found: {self.config.checkpoint_path}")
            raise

    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, List, List]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Tuple of (avg_loss, accuracy, predictions, labels)
        """
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for f_mat, c_mat, labels in pbar:
                # Move data to device
                f_mat = f_mat.to(self.device)
                c_mat = c_mat.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                logits = self.model(f_mat, c_mat)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Get predictions
                _, predicted = torch.max(logits.data, 1)
                
                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * test_correct / test_total:.2f}%'
                })
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * test_correct / test_total
        return avg_loss, accuracy, all_predictions, all_labels
    
    def print_results(
        self,
        checkpoint: Dict,
        test_loss: float,
        test_acc: float,
        predictions: List,
        labels: List
    ):
        
        # Model information
        logger.info("\n" + "=" * 70)
        logger.info("Test Set Evaluation Results")
        logger.info("=" * 70)
        logger.info(f"Checkpoint - Epoch: {checkpoint['epoch'] + 1}")
        logger.info(f"  Training Accuracy:   {checkpoint['train_acc']:.2f}%")
        logger.info(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Test results
        logger.info("\nTest Results:")
        logger.info(f"  Test Loss:     {test_loss:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.2f}%")
        logger.info(f"  Correct:       {sum(p == l for p, l in zip(predictions, labels))}/{len(labels)}")
        
        # Classification report
        logger.info("\nClassification Report:")
        report = classification_report(
            labels,
            predictions,
            target_names=['Class 0', 'Class 1'],
            output_dict=False
        )
        logger.info("\n" + report)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        logger.info(f"\n  True Negatives:  {cm[0, 0]}")
        logger.info(f"  False Positives: {cm[0, 1]}")
        logger.info(f"  False Negatives: {cm[1, 0]}")
        logger.info(f"  True Positives:  {cm[1, 1]}")
        
        # Per-class accuracy
        class_0_acc = (
            cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100
            if (cm[0, 0] + cm[0, 1]) > 0
            else 0
        )
        class_1_acc = (
            cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100
            if (cm[1, 0] + cm[1, 1]) > 0
            else 0
        )
        logger.info("\nPer-Class Accuracy:")
        logger.info(f"  Class 0: {class_0_acc:.2f}%")
        logger.info(f"  Class 1: {class_1_acc:.2f}%")
        
        logger.info("\n" + "=" * 70)
        logger.info("Evaluation Complete!")
        logger.info("=" * 70 + "\n")
        
        # Save confusion matrix as image
        self._save_confusion_matrix(cm)
        
        # Save evaluation metrics as image
        self._save_evaluation_metrics(test_loss, test_acc, class_0_acc, class_1_acc)
    
    def _save_confusion_matrix(self, cm, output_file: str = None):
        """
        Plot and save confusion matrix as an image.
        
        Args:
            cm: Confusion matrix from sklearn
            output_file: Path to save the plot (default: from config)
        """
        if output_file is None:
            output_file = self.config.confusion_matrix_file
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold'},
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'],
            linewidths=2,
            linecolor='black'
        )
        
        # Set labels and title
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
        
        # Add text annotations for TP, TN, FP, FN
        textstr = f'TN: {cm[0,0]}\nFP: {cm[0,1]}\nFN: {cm[1,0]}\nTP: {cm[1,1]}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(
            1.35, 0.5,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=props
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, format='jpg', bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {output_file}")
        plt.close()
    
    def _save_evaluation_metrics(
        self,
        test_loss: float,
        test_acc: float,
        class_0_acc: float,
        class_1_acc: float,
        output_file: str = None
    ):
        """
        Plot and save evaluation metrics as bar chart.
        
        Args:
            test_loss: Test set loss
            test_acc: Test set accuracy
            class_0_acc: Class 0 accuracy
            class_1_acc: Class 1 accuracy
            output_file: Path to save the plot (default: from config)
        """
        if output_file is None:
            output_file = self.config.metrics_file
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].bar(['Test Loss'], [test_loss], color='#FF6B6B', edgecolor='black', linewidth=2, width=0.5)
        axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
        axes[0].set_title('Test Loss', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, max(test_loss * 1.5, 1))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value label on bar
        axes[0].text(
            0, test_loss + test_loss * 0.05,
            f'{test_loss:.4f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
        
        # Plot accuracy
        metrics = ['Test Acc', 'Class 0 Acc', 'Class 1 Acc']
        values = [test_acc, class_0_acc, class_1_acc]
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
        bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=2, width=0.6)
        axes[1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        axes[1].set_title('Test Accuracy Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, value + 2,
                f'{value:.2f}%',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, format='jpg', bbox_inches='tight')
        logger.info(f"✓ Saved evaluation metrics to {output_file}")
        plt.close()
    
    def _save_training_eval_curves(
        self,
        test_loss: float,
        test_acc: float,
        output_file: str = 'training_eval_curves.jpg'
    ):
        """
        Plot and save combined training, validation, and test curves.
        
        Args:
            test_loss: Test set loss
            test_acc: Test set accuracy
            output_file: Path to save the plot (default: training_eval_curves.jpg)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Plot loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2.5)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2.5)
        axes[0].axhline(y=test_loss, color='g', linestyle='--', label='Test Loss', linewidth=2.5)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training, Validation & Test Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Add test loss annotation
        axes[0].text(
            0.98, 0.05,
            f'Test Loss: {test_loss:.4f}',
            transform=axes[0].transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        )
        
        # Plot accuracy curves
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2.5)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=2.5)
        axes[1].axhline(y=test_acc, color='g', linestyle='--', label='Test Accuracy', linewidth=2.5)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training, Validation & Test Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Add test accuracy annotation
        axes[1].text(
            0.98, 0.05,
            f'Test Acc: {test_acc:.2f}%',
            transform=axes[1].transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, format='jpg', bbox_inches='tight')
        logger.info(f"✓ Saved evaluation metrics to {output_file}")
        plt.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Atlas-free Brain Network Transformer (BrainNet)",
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
        '--test_csv',
        type=str,
        default='./data_split/test_df.csv',
        help='Path to test CSV file (default: %(default)s)'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='best_model.pth',
        help='Path to model checkpoint to evaluate (default: %(default)s)'
    )
    
    # Evaluation hyperparameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: %(default)s)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading (default: %(default)s)'
    )
    
    # Misc arguments
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation script."""
    # Parse arguments
    args = parse_arguments()
    
    # Convert args to dictionary for EvaluationConfig
    args_dict = vars(args)
    config = EvaluationConfig(args_dict)
    
    # Setup logging to evaluation directory
    global logger
    logger = setup_logging(config.log_file)
    
    # Log configuration
    logger.info("Evaluation Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Load dataset
    logger.info("Loading test dataset...")
    try:
        test_df = pd.read_csv(config.test_csv)
        test_dataset = MRIDataset(config.data_dir, test_df)
        logger.info(f"Test dataset size: {len(test_dataset)}")
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize evaluator and run evaluation
    evaluator = Evaluator(config)
    
    # Load checkpoint
    checkpoint = evaluator._load_checkpoint()
    logger.info(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate
    test_loss, test_acc, predictions, labels = evaluator.evaluate(test_loader)
    
    # Print results
    evaluator.print_results(checkpoint, test_loss, test_acc, predictions, labels)


if __name__ == "__main__":
    main()

