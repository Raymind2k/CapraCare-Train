"""CapraCare - Grad-CAM for ResNet50."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms, models, datasets

DATA_DIR = Path("CapraCare_Classification_Final")
CHECKPOINT = Path("runs/classification/comparison/ResNet50_best.pt")
RESULTS_JSON = Path("runs/classification/comparison/comparison_results.json")
OUTPUT_DIR = Path("runs/classification/gradcam")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_CLASS = 2


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
        self.model.eval()
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay(img_np, cam, alpha=0.4):
    h = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return (alpha * h + (1 - alpha) * img_np).astype(np.uint8)


def denorm(t):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = t.cpu().numpy().transpose(1, 2, 0)
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def main():
    print("=" * 70)
    print("CapraCare Grad-CAM (ResNet50)")
    print("=" * 70)
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)
    class_names = results["class_names"]
    num_classes = len(class_names)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=tf)
    samples = {i: [] for i in range(num_classes)}
    for idx, (_, lbl) in enumerate(test_ds.samples):
        if len(samples[lbl]) < SAMPLES_PER_CLASS:
            samples[lbl].append(idx)
    fig, axes = plt.subplots(num_classes, SAMPLES_PER_CLASS * 2, figsize=(SAMPLES_PER_CLASS * 4, num_classes * 2))
    for ci in range(num_classes):
        for col, si in enumerate(samples[ci]):
            img_t, true_lbl = test_ds[si]
            x = img_t.unsqueeze(0).to(DEVICE)
            x.requires_grad_(True)
            with torch.no_grad():
                pred = model(x).argmax(1).item()
            cam = gradcam.generate(x, pred)
            img_np = denorm(img_t)
            ov = overlay(img_np, cam)
            axes[ci, col * 2].imshow(img_np)
            axes[ci, col * 2].set_title(class_names[true_lbl][:15], fontsize=8)
            axes[ci, col * 2].axis("off")
            axes[ci, col * 2 + 1].imshow(ov)
            axes[ci, col * 2 + 1].set_title("Pred: " + class_names[pred][:15], fontsize=8)
            axes[ci, col * 2 + 1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradcam_summary_grid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'gradcam_summary_grid.png'}")


if __name__ == "__main__":
    main()
