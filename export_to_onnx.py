import torch
from torchvision import models

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "landmark_resnet18_final.pth"
ONNX_PATH = "landmark_resnet18.onnx"
NUM_CLASSES = 3103   # must match training
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(weights=None)  # no need for pretrained weights
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load your trained weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
model = model.to(DEVICE)

# -----------------------------
# Create dummy input
# -----------------------------
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

# -----------------------------
# Export to ONNX
# -----------------------------
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
