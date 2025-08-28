import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from collections import Counter

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../dataset/images"
TRAIN_CSV = "../dataset/train.csv"
VAL_CSV = "../dataset/val.csv"
BATCH_SIZE = 16
NUM_EPOCHS = 25        # Increased slightly
LEARNING_RATE = 5e-4   # Slightly reduced for stability
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
WARMUP_EPOCHS = 2
PATIENCE = 7
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")

# -----------------------------
# Dataset (unchanged to maintain compatibility)
# -----------------------------
class LandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        label = int(row['landmark_id'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Enhanced Transforms with stronger augmentation
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# Data Loaders
# -----------------------------
train_dataset = LandmarkDataset(TRAIN_CSV, DATA_DIR, transform=train_transforms)
val_dataset = LandmarkDataset(VAL_CSV, DATA_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# Calculate class weights for imbalanced datasets
# -----------------------------
def calculate_class_weights(df):
    class_counts = df['landmark_id'].value_counts().sort_index()
    total_samples = len(df)
    num_classes = len(class_counts)
    
    # Calculate weights inversely proportional to class frequency
    weights = []
    for class_id in sorted(class_counts.index):
        weight = total_samples / (num_classes * class_counts[class_id])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

class_weights = calculate_class_weights(train_dataset.df).to(DEVICE)
print(f"Using class weights for {len(class_weights)} classes")

# -----------------------------
# Model (EfficientNet-V2-S for better performance)
# -----------------------------
num_classes = train_dataset.df['landmark_id'].nunique()
print(f"Number of classes: {num_classes}")

# Use EfficientNet-V2-S (better than B3)
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

# Loss function with label smoothing and class weights
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# Optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Mixed precision training
scaler = GradScaler() if DEVICE == "cuda" else None

# Warmup scheduler
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS * len(train_loader)
) if WARMUP_EPOCHS > 0 else None

# -----------------------------
# Resume from last checkpoint if exists (maintaining compatibility)
# -----------------------------
def load_checkpoint():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        return 0, 0, 0.0, {}
    
    # Sort by the number in filename
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    last_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    
    checkpoint = torch.load(last_checkpoint, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        start_epoch = checkpoint.get('epoch', 0)
        total_processed = checkpoint.get('total_processed', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resuming from checkpoint: {last_checkpoint}")
        print(f"Epoch: {start_epoch}, Total processed: {total_processed}, Best val acc: {best_val_acc:.4f}")
        return start_epoch, total_processed, best_val_acc, checkpoint
    else:
        # Old format compatibility
        model.load_state_dict(checkpoint)
        total_processed = int(checkpoint_files[-1].split('_')[-1].split('.')[0])
        print(f"Resuming from old format checkpoint: {last_checkpoint}")
        return 0, total_processed, 0.0, {}

start_epoch, total_processed, best_val_acc, checkpoint_data = load_checkpoint()
patience_counter = checkpoint_data.get('patience_counter', 0)

if start_epoch == 0 and total_processed == 0:
    print("No checkpoint found, starting fresh.")

# -----------------------------
# Training metrics tracking
# -----------------------------
def save_checkpoint(epoch, total_processed, val_acc, is_best=False):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_processed': total_processed,
        'best_val_acc': best_val_acc,
        'patience_counter': patience_counter,
        'num_classes': num_classes
    }
    
    # Regular checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{total_processed}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Best model checkpoint
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
    
    return checkpoint_path

# -----------------------------
# Training Loop with all improvements
# -----------------------------
print("Starting training...")
training_history = []

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training phase
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
        
        # Warmup scheduler
        if warmup_scheduler is not None and epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        
        # Statistics
        running_loss += loss.item() * imgs.size(0)
        total_processed += imgs.size(0)
        
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        
        # Save checkpoint every 1000 images (maintaining compatibility)
        if total_processed % 1000 < BATCH_SIZE:
            checkpoint_path = save_checkpoint(epoch, total_processed, best_val_acc)
            print(f"Checkpoint saved at {total_processed} images: {checkpoint_path}")
    
    # Step scheduler after warmup
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()
    
    epoch_loss = running_loss / len(train_dataset)
    train_acc = correct_train / total_train
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr:.6f}")

    # -----------------------------
    # Validation with early stopping
    # -----------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            if scaler is not None:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_acc = correct / total
    val_loss_avg = val_loss / len(val_dataset)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_acc:.4f}")
    
    # Save training history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_acc': train_acc,
        'val_loss': val_loss_avg,
        'val_acc': val_acc,
        'lr': current_lr
    })
    
    # Early stopping and best model saving
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Save checkpoint
    save_checkpoint(epoch, total_processed, val_acc, is_best)
    
    print(f"Best Val Acc: {best_val_acc:.4f}, Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered after {patience_counter} epochs without improvement")
        break

# -----------------------------
# Save final model (maintaining original filename for compatibility)
# -----------------------------
torch.save(model.state_dict(), "landmark_resnet18_final.pth")
print("Training complete. Final model saved as 'landmark_resnet18_final.pth'")

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)

# Save model info for inference
model_info = {
    'num_classes': num_classes,
    'img_size': IMG_SIZE,
    'model_architecture': 'efficientnet_v2_s',
    'best_val_accuracy': best_val_acc,
    'total_epochs_trained': epoch + 1,
    'total_images_processed': total_processed
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nTraining Summary:")
print(f"- Best validation accuracy: {best_val_acc:.4f}")
print(f"- Total epochs trained: {epoch + 1}")
print(f"- Total images processed: {total_processed}")
print(f"- Model architecture: EfficientNet-V2-S")
print(f"- Training history saved to: training_history.json")
print(f"- Model info saved to: model_info.json")