import torch
from torchvision import models
import json
import os

# -----------------------------
# Config - Auto-detect from training
# -----------------------------
MODEL_PATH = "landmark_resnet18_final.pth"
ONNX_PATH = "landmark_resnet18.onnx"
MODEL_INFO_PATH = "model_info.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model configuration
# -----------------------------
def load_model_config():
    """Load model configuration from training output"""
    if os.path.exists(MODEL_INFO_PATH):
        print(f"Loading model configuration from {MODEL_INFO_PATH}")
        with open(MODEL_INFO_PATH, 'r') as f:
            config = json.load(f)
        return config
    else:
        print("Warning: model_info.json not found. Using fallback detection...")
        # Fallback: Try to detect from model file
        return detect_model_config()

def detect_model_config():
    """Fallback method to detect model configuration"""
    # Load the state dict to inspect the model
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Check if it's the new format (dict with metadata) or old format (just state dict)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        # New checkpoint format
        actual_state_dict = state_dict['model_state_dict']
        num_classes = state_dict.get('num_classes', None)
        
        # Detect architecture from layer names
        if 'classifier.1.weight' in actual_state_dict:
            if 'features.0.0.weight' in actual_state_dict:
                # EfficientNet architecture
                model_arch = 'efficientnet_v2_s'
            else:
                model_arch = 'efficientnet_b3'
        elif 'fc.weight' in actual_state_dict:
            model_arch = 'resnet18'
        else:
            raise ValueError("Unable to detect model architecture")
            
        if num_classes is None:
            # Detect from final layer
            if model_arch in ['efficientnet_v2_s', 'efficientnet_b3']:
                num_classes = actual_state_dict['classifier.1.weight'].shape[0]
            else:
                num_classes = actual_state_dict['fc.weight'].shape[0]
        
        return {
            'model_architecture': model_arch,
            'num_classes': num_classes,
            'img_size': 224
        }
    else:
        # Old format - just the state dict
        if 'classifier.1.weight' in state_dict:
            if 'features.0.0.weight' in state_dict:
                model_arch = 'efficientnet_v2_s'
                num_classes = state_dict['classifier.1.weight'].shape[0]
            else:
                model_arch = 'efficientnet_b3'
                num_classes = state_dict['classifier.1.weight'].shape[0]
        elif 'fc.weight' in state_dict:
            model_arch = 'resnet18'
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            raise ValueError("Unable to detect model architecture from state dict")
            
        return {
            'model_architecture': model_arch,
            'num_classes': num_classes,
            'img_size': 224
        }

config = load_model_config()
NUM_CLASSES = config['num_classes']
IMG_SIZE = config['img_size']
MODEL_ARCH = config['model_architecture']

print(f"Detected configuration:")
print(f"  - Architecture: {MODEL_ARCH}")
print(f"  - Number of classes: {NUM_CLASSES}")
print(f"  - Image size: {IMG_SIZE}")

# -----------------------------
# Load Model with correct architecture
# -----------------------------
def create_model(architecture, num_classes):
    """Create model with the correct architecture"""
    if architecture == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model

model = create_model(MODEL_ARCH, NUM_CLASSES)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # New checkpoint format
    state_dict = checkpoint['model_state_dict']
    print("Loading from new checkpoint format")
else:
    # Old format
    state_dict = checkpoint
    print("Loading from old checkpoint format")

model.load_state_dict(state_dict)
model.eval()
model = model.to(DEVICE)

print(f"Model loaded successfully on {DEVICE}")

# -----------------------------
# Create dummy input
# -----------------------------
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

# Test the model with dummy input
print("Testing model with dummy input...")
with torch.no_grad():
    test_output = model(dummy_input)
    print(f"Model output shape: {test_output.shape}")
    assert test_output.shape[1] == NUM_CLASSES, f"Output classes mismatch: got {test_output.shape[1]}, expected {NUM_CLASSES}"

# -----------------------------
# Export to ONNX
# -----------------------------
print(f"Exporting to ONNX format: {ONNX_PATH}")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX model exported successfully as {ONNX_PATH}")

# -----------------------------
# Verify ONNX model
# -----------------------------
try:
    import onnx
    import onnxruntime as ort
    
    # Verify ONNX model
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
    
    # Test ONNX runtime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    
    # Test with dummy input
    dummy_input_np = dummy_input.cpu().numpy()
    onnx_output = session.run(None, {'input': dummy_input_np})[0]
    
    print(f"ONNX Runtime test passed! Output shape: {onnx_output.shape}")
    
    # Compare outputs
    torch_output = test_output.cpu().numpy()
    max_diff = abs(torch_output - onnx_output).max()
    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX export successful - outputs match!")
    else:
        print("⚠️  Warning: Outputs differ significantly")
        
except ImportError:
    print("onnx or onnxruntime not installed. Skipping verification.")
except Exception as e:
    print(f"ONNX verification failed: {e}")

# -----------------------------
# Save export info
# -----------------------------
export_info = {
    'onnx_path': ONNX_PATH,
    'model_architecture': MODEL_ARCH,
    'num_classes': NUM_CLASSES,
    'img_size': IMG_SIZE,
    'input_name': 'input',
    'output_name': 'output',
    'opset_version': 17
}

with open('export_info.json', 'w') as f:
    json.dump(export_info, f, indent=2)

print("Export information saved to export_info.json")