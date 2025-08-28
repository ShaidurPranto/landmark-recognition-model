import os
import pandas as pd
import torch
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# -----------------------------
# Config - Auto-detect from export
# -----------------------------
DATA_DIR = "./dataset/images"   # image folder
TEST_CSV = "./dataset/test.csv" # CSV file with filename, landmark_id
ONNX_PATH = "landmark_resnet18.onnx"
EXPORT_INFO_PATH = "export_info.json"
BATCH_SIZE = 32  # Increased for better performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load export configuration
# -----------------------------
def load_export_config():
    """Load export configuration to ensure compatibility"""
    if os.path.exists(EXPORT_INFO_PATH):
        print(f"Loading export configuration from {EXPORT_INFO_PATH}")
        with open(EXPORT_INFO_PATH, 'r') as f:
            config = json.load(f)
        return config
    else:
        print("Warning: export_info.json not found. Using default values...")
        return {
            'img_size': 224,
            'input_name': 'input',
            'output_name': 'output',
            'onnx_path': ONNX_PATH
        }

config = load_export_config()
IMG_SIZE = config['img_size']
INPUT_NAME = config.get('input_name', 'input')
OUTPUT_NAME = config.get('output_name', 'output')
ONNX_MODEL_PATH = config.get('onnx_path', ONNX_PATH)

print(f"Configuration loaded:")
print(f"  - ONNX model: {ONNX_MODEL_PATH}")
print(f"  - Image size: {IMG_SIZE}")
print(f"  - Input name: {INPUT_NAME}")
print(f"  - Output name: {OUTPUT_NAME}")
print(f"  - Batch size: {BATCH_SIZE}")

# -----------------------------
# Dataset for ONNX Testing (unchanged for compatibility)
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
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Transforms (same as training validation transforms)
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# Data Loader
# -----------------------------
print(f"Loading test dataset from {TEST_CSV}")
test_dataset = LandmarkDataset(TEST_CSV, DATA_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Test dataset loaded: {len(test_dataset)} images")
print(f"Number of batches: {len(test_loader)}")

# -----------------------------
# Load ONNX Model
# -----------------------------
print(f"Loading ONNX model: {ONNX_MODEL_PATH}")

# Set up providers based on device
if DEVICE == "cuda" and torch.cuda.is_available():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Using CUDA execution provider")
else:
    providers = ["CPUExecutionProvider"]
    print("Using CPU execution provider")

try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    print("ONNX Runtime session created successfully")
    
    # Get actual input/output names from the model
    actual_input_name = session.get_inputs()[0].name
    actual_output_name = session.get_outputs()[0].name
    
    print(f"Model input name: {actual_input_name}")
    print(f"Model output name: {actual_output_name}")
    print(f"Model input shape: {session.get_inputs()[0].shape}")
    print(f"Model output shape: {session.get_outputs()[0].shape}")
    
    # Use actual names if they differ from config
    if actual_input_name != INPUT_NAME:
        print(f"Using actual input name: {actual_input_name} (config had: {INPUT_NAME})")
        INPUT_NAME = actual_input_name
    
    if actual_output_name != OUTPUT_NAME:
        print(f"Using actual output name: {actual_output_name} (config had: {OUTPUT_NAME})")
        OUTPUT_NAME = actual_output_name
        
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print("Please make sure the ONNX export completed successfully")
    exit(1)

# -----------------------------
# Evaluation with detailed metrics
# -----------------------------
print("\nStarting evaluation...")

correct = 0
total = 0
class_correct = {}
class_total = {}
all_predictions = []
all_labels = []

for batch_idx, (imgs, labels) in enumerate(test_loader):
    try:
        # Convert to numpy for ONNX Runtime
        imgs_np = imgs.numpy()
        labels_np = labels.numpy()
        
        # Run inference
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: imgs_np})[0]
        preds = outputs.argmax(axis=1)
        
        # Update statistics
        batch_correct = (preds == labels_np).sum()
        correct += batch_correct
        total += labels.size(0)
        
        # Per-class statistics
        for i in range(len(labels_np)):
            label = labels_np[i]
            pred = preds[i]
            
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
                
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
        
        # Store for detailed analysis
        all_predictions.extend(preds.tolist())
        all_labels.extend(labels_np.tolist())
        
        # Progress update
        if (batch_idx + 1) % 10 == 0:
            current_acc = correct / total
            print(f"Batch {batch_idx + 1}/{len(test_loader)}: Running accuracy: {current_acc:.4f}")
            
    except Exception as e:
        print(f"Error processing batch {batch_idx}: {e}")
        continue

# -----------------------------
# Calculate final metrics
# -----------------------------
if total > 0:
    accuracy = correct / total
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Error Rate: {(1-accuracy):.4f}")
    
    # Top-5 accuracy if there are enough classes
    unique_classes = len(set(all_labels))
    print(f"Number of classes in test set: {unique_classes}")
    
    if unique_classes >= 5:
        # Calculate top-5 accuracy
        top5_correct = 0
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs_np = imgs.numpy()
            labels_np = labels.numpy()
            outputs = session.run([OUTPUT_NAME], {INPUT_NAME: imgs_np})[0]
            
            # Get top-5 predictions
            top5_preds = np.argsort(outputs, axis=1)[:, -5:]
            
            for i in range(len(labels_np)):
                if labels_np[i] in top5_preds[i]:
                    top5_correct += 1
                    
        top5_accuracy = top5_correct / total
        print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_correct}/{total})")
    
    # Per-class accuracy for classes with enough samples
    print(f"\nPer-class accuracy (classes with >5 samples):")
    print("-" * 40)
    sorted_classes = sorted(class_total.items(), key=lambda x: x[1], reverse=True)
    
    for class_id, class_count in sorted_classes[:20]:  # Show top 20 classes
        if class_count > 5:
            class_acc = class_correct[class_id] / class_count
            print(f"Class {class_id}: {class_acc:.3f} ({class_correct[class_id]}/{class_count})")
    
    # Save detailed results
    results = {
        'overall_accuracy': float(accuracy),
        'total_samples': int(total),
        'correct_predictions': int(correct),
        'num_classes_tested': unique_classes,
        'per_class_accuracy': {
            str(k): float(class_correct[k] / class_total[k]) 
            for k in class_total.keys() if class_total[k] > 0
        },
        'batch_size': BATCH_SIZE,
        'model_path': ONNX_MODEL_PATH
    }
    
    if 'top5_accuracy' in locals():
        results['top5_accuracy'] = float(top5_accuracy)
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: evaluation_results.json")
    
else:
    print("No samples were processed successfully!")

print(f"\nEvaluation completed!")

# -----------------------------
# Memory cleanup
# -----------------------------
del session
if 'outputs' in locals():
    del outputs
if 'imgs_np' in locals():
    del imgs_np