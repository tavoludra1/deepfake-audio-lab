"""
Mission-Critical Trainer Engine.
Implements Automatic Mixed Precision (AMP) and Gradient Clipping.
Standard: Google DeepMind Research Engineering.
"""
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import logging
from ..infrastructure.numerics.stability import NumericalGuard

class ResearchTrainer:
    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scaler = GradScaler() # Optimization for Colab Pro GPUs
        self.guard = NumericalGuard()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mission Critical: Autocast for memory efficiency and speed
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Check for NaNs before backward pass
                self.guard.validate_tensor(loss, f"Loss_Batch_{batch_idx}")

            # Scales loss and calls backward() to create scaled gradients
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping: Prevents exploding gradients (Standard in DeepMind)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        return correct / len(dataloader.dataset)
