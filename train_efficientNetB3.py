import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./dataset/images"
TRAIN_CSV = "./dataset/train.csv"
VAL_CSV = "./dataset/val.csv"
BATCH_SIZE = 16
NUM_EPOCHS = 20        # increase epochs for better accuracy
LEARNING_RATE = 1e-3
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------
# Dataset
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
# Transforms (strong augmentation for training)
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Data Loaders
# -----------------------------
train_dataset = LandmarkDataset(TRAIN_CSV, DATA_DIR, transform=train_transforms)
val_dataset = LandmarkDataset(VAL_CSV, DATA_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Model (EfficientNet-B3)
# -----------------------------
num_classes = train_dataset.df['landmark_id'].nunique()
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Resume from last checkpoint if exists
# -----------------------------
checkpoint_files = sorted(os.listdir(CHECKPOINT_DIR))
start_epoch = 0
total_processed = 0  # absolute counter

if checkpoint_files:
    last_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    model.load_state_dict(torch.load(last_checkpoint))
    print(f"Resuming from checkpoint: {last_checkpoint}")
    total_processed = int(checkpoint_files[-1].split('_')[-1].split('.')[0])
else:
    print("No checkpoint found, starting fresh.")

# -----------------------------
# Training Loop with proper checkpointing
# -----------------------------
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total_processed += imgs.size(0)

        # Save checkpoint every 1000 images
        if total_processed % 1000 < BATCH_SIZE:  
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{total_processed}.pth")
            if not os.path.exists(checkpoint_path):  
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {total_processed} images: {checkpoint_path}")

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val Accuracy: {val_acc:.4f}")

# -----------------------------
# Save final model
# -----------------------------
torch.save(model.state_dict(), "landmark_resnet18_final.pth")
print("Training complete. Final model saved as 'landmark_resnet18_final.pth'")
