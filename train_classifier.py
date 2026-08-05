"""
CapraCare - Train single EfficientNet-B0 classifier.
Quick proof-of-concept training. Use train_comparison.py for the full 4-model run.
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
OUTPUT_DIR = Path("runs/classification/single")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42


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
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


from labnet import LABNet

def build_model(num_classes):
    return LABNet(num_classes)


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


def main():
    torch.manual_seed(SEED)
    print("=" * 70)
    print("CapraCare - DLAB-Net Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = get_transforms()
    
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)
    
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = build_model(num_classes).to(DEVICE)
    class_weights = compute_class_weights(train_ds, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\nTraining for {EPOCHS} epochs...")
    start = time.time()
    
    for epoch in range(EPOCHS):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl, va, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  TrainLoss={tl:.4f} TrainAcc={ta:.2f}%  ValLoss={vl:.4f} ValAcc={va:.2f}%")
        
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
    
    elapsed = (time.time() - start) / 60
    print(f"\nTraining complete in {elapsed:.1f} min. Best val acc: {best_val_acc:.2f}%")
    
    # Test
    model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt"))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        "model_name": "DLAB-Net",
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_preds": preds,
        "test_labels": labels,
        "class_names": class_names,
        "history": history,
        "training_time_min": elapsed,
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()