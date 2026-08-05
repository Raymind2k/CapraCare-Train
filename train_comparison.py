"""
CapraCare - Train 4 architectures with identical settings for fair comparison.
Models: EfficientNet-B0, ResNet50, MobileNetV2, DenseNet121
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Config
DATA_DIR = Path("CapraCare_Classification_Final")
OUTPUT_DIR = Path("runs/classification/comparison")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def compute_class_weights(dataset, num_classes):
    counts = [0] * num_classes
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    return torch.tensor(
        [total / (num_classes * c) if c > 0 else 0.0 for c in counts],
        dtype=torch.float32
    )


def build_model(name, num_classes):
    """Build model with ImageNet pretrained weights, replace final layer."""
    if name == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        head_name = "classifier"
    elif name == "ResNet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        head_name = "fc"
    elif name == "MobileNetV2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        head_name = "classifier"
    elif name == "DenseNet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
        head_name = "classifier"
    else:
        raise ValueError(f"Unknown model: {name}")
    return m, head_name


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def train_single_model(name, train_loader, val_loader, test_loader, num_classes, class_weights, class_names):
    print("\n" + "=" * 70)
    print(f"Training {name}")
    print("=" * 70)
    
    set_seed(SEED)
    model, head_name = build_model(name, num_classes)
    model = model.to(DEVICE)
    n_params = count_params(model)
    print(f"Parameters: {n_params:.2f}M")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    head_params = [p for n, p in model.named_parameters() if n.startswith(head_name)]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith(head_name)]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_path = OUTPUT_DIR / f"{name}_best.pt"
    
    start = time.time()
    for epoch in range(EPOCHS):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl, va, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        
        marker = ""
        if va > best_val_acc:
            best_val_acc = va
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            marker = "  ★ best"
        
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  TL={tl:.3f} TA={ta:5.2f}%  VL={vl:.3f} VA={va:5.2f}%{marker}")
    
    elapsed = (time.time() - start) / 60
    
    # Final test eval on best checkpoint
    model.load_state_dict(torch.load(ckpt_path))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n  Best epoch: {best_epoch}, Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Training time: {elapsed:.1f} min")
    
    return {
        "model_name": name,
        "num_params_millions": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "history": history,
        "training_time_min": elapsed,
    }


def main():
    print("=" * 70)
    print("CapraCare - 4-Architecture Comparison")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)
    
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    class_weights = compute_class_weights(train_ds, num_classes).to(DEVICE)
    
    model_names = ["EfficientNet-B0", "ResNet50", "MobileNetV2", "DenseNet121"]
    all_results = []
    
    overall_start = time.time()
    for name in model_names:
        result = train_single_model(name, train_loader, val_loader, test_loader,
                                     num_classes, class_weights, class_names)
        all_results.append(result)
    
    total_time = (time.time() - overall_start) / 60
    
    # Save combined results
    combined = {
        "class_names": class_names,
        "num_classes": num_classes,
        "config": {
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "seed": SEED,
        },
        "results": all_results,
        "total_time_min": total_time,
    }
    
    with open(OUTPUT_DIR / "comparison_results.json", 'w') as f:
        json.dump(combined, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Params(M)':>10} {'BestEpoch':>10} {'ValAcc':>8} {'TestAcc':>8} {'Time(min)':>10}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: -x["test_acc"]):
        print(f"{r['model_name']:<20} {r['num_params_millions']:>10.2f} {r['best_epoch']:>10} "
              f"{r['best_val_acc']:>7.2f}% {r['test_acc']:>7.2f}% {r['training_time_min']:>10.1f}")
    print(f"\nTotal training time: {total_time:.1f} min")
    print(f"Results saved to: {OUTPUT_DIR / 'comparison_results.json'}")


if __name__ == '__main__':
    main()