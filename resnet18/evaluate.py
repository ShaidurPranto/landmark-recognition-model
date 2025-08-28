import os
import pandas as pd
import torch
import onnxruntime as ort
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Fix PIL issues
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../dataset/images"   # image folder
TEST_CSV = "../dataset/test.csv" # CSV file with filename, landmark_id
ONNX_PATH = "landmark_resnet18.onnx"
BATCH_SIZE = 16
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset for ONNX Testing
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
# Transforms
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------
# Data Loader
# -----------------------------
test_dataset = LandmarkDataset(TEST_CSV, DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load ONNX Model
# -----------------------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_PATH, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# Evaluation
# -----------------------------
correct = 0
total = 0

for imgs, labels in test_loader:
    imgs_np = imgs.cpu().numpy().astype("float32")  # ✅ ensure float32
    outputs = session.run([output_name], {input_name: imgs_np})[0]
    preds = outputs.argmax(axis=1)

    correct += (preds == labels.numpy()).sum().item()
    total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")  # ✅ show percentage
