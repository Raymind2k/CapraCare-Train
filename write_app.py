content = '''"""CapraCare - Streamlit Demo App"""
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import streamlit as st
from PIL import Image
from torchvision import transforms, models

from treatment_recommendations import get_recommendation

# ============================================================
# Configuration
# ============================================================
CHECKPOINT = Path("runs/classification/comparison/ResNet50_best.pt")
RESULTS_JSON = Path("runs/classification/comparison/comparison_results.json")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="CapraCare - Goat Disease Detector",
    page_icon="🐐",
    layout="wide",
)

# ============================================================
# Model loading (cached)
# ============================================================
@st.cache_resource
def load_model_and_classes():
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)
    class_names = results["class_names"]
    num_classes = len(class_names)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model, class_names


# ============================================================
# Grad-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(img_np, cam, alpha=0.4):
    h = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return (alpha * h + (1 - alpha) * img_np).astype(np.uint8)


# ============================================================
# Image preprocessing
# ============================================================
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, img_pil, class_names):
    img_t = tf(img_pil).unsqueeze(0).to(DEVICE)
    img_t.requires_grad_(True)
    
    # Forward pass with gradients enabled for Grad-CAM
    gradcam = GradCAM(model, model.layer4[-1])
    
    # Get prediction
    with torch.no_grad():
        out = model(img_t)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
    
    # Generate Grad-CAM
    cam = gradcam.generate(img_t, pred_idx)
    
    # Build overlay
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized)
    overlay = overlay_heatmap(img_np, cam)
    
    return pred_idx, pred_conf, probs, overlay, img_np


# ============================================================
# UI
# ============================================================
st.title("🐐 CapraCare — Goat Skin Disease Detector")
st.markdown("Upload a photo of a goat skin lesion to get an AI-powered diagnosis and treatment recommendation.")

with st.sidebar:
    st.header("About")
    st.markdown(\"\"\"
**Model:** ResNet50 (transfer learning)  
**Test Accuracy:** 97.97%  
**Classes:** 8 goat skin diseases

**Disclaimer:** This is a research prototype. Always consult a veterinarian for actual diagnosis and treatment.
\"\"\")
    st.divider()
    st.subheader("Supported diseases")
    st.markdown(\"\"\"
- Parasitic Mange
- Myiasis (Maggot infestation)
- Skin Infection: Abscess
- Skin Infection: Dermatitis
- Skin Infection: Dermatophilosis
- Viral: Orf
- Viral: PPR
- Wounds: Footrot
\"\"\")

# Load model
with st.spinner("Loading model..."):
    model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Upload a goat skin image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    
    with st.spinner("Analyzing image..."):
        pred_idx, pred_conf, probs, overlay, img_np = predict(model, img_pil, class_names)
    
    pred_class = class_names[pred_idx]
    
    # Show original + Grad-CAM
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_np, use_container_width=True)
    with col2:
        st.subheader("AI Attention Map (Grad-CAM)")
        st.image(overlay, use_container_width=True)
        st.caption("Red/yellow regions = areas the model focused on for its prediction")
    
    # Prediction header
    st.divider()
    pretty_name = pred_class.replace("_", " ").title()
    if pred_conf >= 0.80:
        st.success(f"### Prediction: **{pretty_name}**  ({pred_conf:.1%} confidence)")
    elif pred_conf >= 0.60:
        st.warning(f"### Prediction: **{pretty_name}**  ({pred_conf:.1%} confidence) — moderate confidence")
    else:
        st.error(f"### Prediction: **{pretty_name}**  ({pred_conf:.1%} confidence) — LOW confidence, consult a vet")
    
    # Top-3 probabilities
    st.subheader("All class probabilities")
    top3_idx = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(top3_idx):
        cn = class_names[idx].replace("_", " ").title()
        st.progress(float(probs[idx]), text=f"{cn}: {probs[idx]:.1%}")
    
    # Treatment recommendation
    st.divider()
    st.subheader("📋 Treatment Recommendation")
    rec = get_recommendation(pred_class, confidence=pred_conf)
    st.text(rec)

else:
    st.info("👆 Upload an image to begin.")
    st.markdown("**Tip:** Use clear, close-up photos of the affected skin area for best results.")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ app.py has been created.")
print(f"   File size: {len(content)} bytes")