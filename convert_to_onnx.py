"""
Convert ResNet50 PyTorch checkpoint → ONNX for on-device mobile inference.
Verifies the ONNX model matches the PyTorch model on a real test image.

Note: ResNet50_best.pt is a bare state_dict — class names are inferred from
the alphabetically sorted test folder structure (ImageFolder convention).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import onnx
import onnxruntime as ort

CHECKPOINT = Path("runs/classification/comparison/ResNet50_best.pt")
OUTPUT_ONNX = Path("capracare_resnet50.onnx")
TEST_DIR = Path("CapraCare_Classification_Final/test")

IMG_SIZE = 224


def get_class_names(test_dir: Path):
    """Mirror ImageFolder's alphabetical class indexing."""
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir.resolve()}")
    return sorted(p.name for p in test_dir.iterdir() if p.is_dir())


def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def find_test_image(test_dir: Path):
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return img, cls_dir.name
    return None, None


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT.resolve()}")

    class_names = get_class_names(TEST_DIR)
    print(f"Classes ({len(class_names)}): {class_names}")

    print("\n" + "=" * 60)
    print("Loading ResNet50 weights...")
    print("=" * 60)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_resnet50(len(class_names))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Loaded state_dict with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    print("=" * 60)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        OUTPUT_ONNX,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✓ ONNX file written: {OUTPUT_ONNX.resolve()}")
    print(f"  Size: {OUTPUT_ONNX.stat().st_size / 1e6:.2f} MB")

    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX structural check passed")

    print("\n" + "=" * 60)
    print("Verifying PyTorch ↔ ONNX parity...")
    print("=" * 60)
    img_path, true_class = find_test_image(TEST_DIR)
    if img_path is None:
        print("⚠ No test image found, skipping parity check.")
        return

    print(f"Test image: {img_path.name} (true class: {true_class})")
    tf = get_transform()
    pil = Image.open(img_path).convert("RGB")
    x = tf(pil).unsqueeze(0)

    with torch.no_grad():
        torch_logits = model(x).numpy()
    torch_probs = F.softmax(torch.from_numpy(torch_logits), dim=1).numpy()[0]
    torch_pred_idx = int(np.argmax(torch_probs))

    sess = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"input": x.numpy()})[0]
    onnx_probs = F.softmax(torch.from_numpy(onnx_logits), dim=1).numpy()[0]
    onnx_pred_idx = int(np.argmax(onnx_probs))

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))

    print(f"\nPyTorch prediction: {class_names[torch_pred_idx]} "
          f"({torch_probs[torch_pred_idx]*100:.2f}%)")
    print(f"ONNX prediction:    {class_names[onnx_pred_idx]} "
          f"({onnx_probs[onnx_pred_idx]*100:.2f}%)")
    print(f"Max logit difference: {max_diff:.6f}")

    if torch_pred_idx == onnx_pred_idx and max_diff < 1e-3:
        print("\n✓ SUCCESS — ONNX model matches PyTorch exactly.")
    elif torch_pred_idx == onnx_pred_idx:
        print(f"\n⚠ Predictions match but logits differ by {max_diff:.4f} — usually fine.")
    else:
        print("\n✗ MISMATCH — predictions differ. Investigate before deploying.")


if __name__ == "__main__":
    main()