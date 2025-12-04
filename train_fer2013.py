"""
FER-2013 Facial Expression Recognition 

- Two model options:
    1) simple_cnn  (default): lightweight CNN trained from scratch on 48x48 grayscale
    2) resnet18    : ResNet-18 backbone (by default starts from random init to avoid internet)
                    can try --pretrained to attempt loading ImageNet weights

- Includes:
  * Reproducibility seed
  * Data augmentation
  * Class-imbalance aware loss weights
  * Stratified train/val split
  * Mixed precision training
  * Early stopping + ReduceLROnPlateau scheduler
  * Confusion matrix + classification report (saved to disk)
  * Model checkpointing (best + last)

# default (simple CNN, 48x48 grayscale)
python train_fer2013.py

# hyper-params
python train_fer2013.py --epochs 25 --batch-size 128 --lr 3e-4 --val-split 0.1
"""

import os         
import json       
import math      
import random      
import argparse     
from dataclasses import dataclass  # structured configuration container
from typing import Tuple, List    
import torch                                  # tensors and autograd
import torch.nn as nn                         # neural network layers
import torch.optim as optim                   # optimizers 
from torch.utils.data import DataLoader, Subset # data batching utilities
from torch.cuda.amp import autocast, GradScaler # mixed-precision acceleration
from torchvision import transforms, models     # image transforms and model zoo
from torchvision.datasets import ImageFolder   # folder-based dataset wrapper
import numpy as np                                          
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt                                 

try:
    from tqdm import tqdm  # for progress bars 
except ImportError:
    tqdm = lambda x, **kw: x  


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)                  # Python RNG
    np.random.seed(seed)               # NumPy RNG
    torch.manual_seed(seed)            # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)   # PyTorch GPU RNG
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    data_root: str = "./archive"       # dataset root 
    train_dirname: str = "train"       # subfolder containing training images
    # training hyperparameters
    model: str = "simple_cnn"          # "simple_cnn" or "resnet18"
    img_size: int = 48                 # default input size; 48 for simple CNN
    batch_size: int = 128              # batch size
    epochs: int = 20                   # total epochs
    lr: float = 3e-4                   # learning rate
    weight_decay: float = 1e-4         # L2 regularization
    val_split: float = 0.1             # fraction of train used for validation
    seed: int = 42                     # RNG seed
    num_workers: int = 2               # DataLoader workers 
    pretrained: bool = False           # try using pretrained weights for resnet18
    use_amp: bool = True               # mixed precision training
    patience: int = 7                  # early stopping patience
    grad_clip: float = 1.0             # gradient clipping max-norm (0 to disable)
    no_class_weights: bool = False     # disable class-imbalance weighting if True
    balance_sampler: bool = False      # (kept False) (using loss weights is simpler/robust)
    save_dir: str = "./outputs"        # directory to save models & reports

def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """ For simple_cnn we use GRAYSCALE (1 channel) and small 48x48 crops.
    for resnet18 we use RGB (3 channels) and 224x224 typical ImageNet-like transforms.
    """
    if cfg.model == "resnet18":
        # RGB, larger input; augment with common ImageNet-style policies
        train_tfms = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),     # resize to target size
            transforms.RandomHorizontalFlip(p=0.5),              # mirror faces for invariance
            transforms.RandomRotation(10),                       # slight rotations
            transforms.ColorJitter(brightness=0.1, contrast=0.1),# mild color jitter
            transforms.ToTensor(),                               # HWC [0..255] -> CHW [0..1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],     # standard ImageNet normalization
                                 std=[0.229, 0.224, 0.225]),])
        val_tfms = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
    else:
        # simple_cnn: grayscale pipeline for 48x48 FER images
        train_tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),         # convert to single channel
            transforms.Resize((cfg.img_size, cfg.img_size)),     # ensure 48x48
            transforms.RandomHorizontalFlip(p=0.5),              # augment with flips
            transforms.RandomRotation(10),                       # small rotations
            transforms.ToTensor(),                               # to tensor [0..1]
            transforms.Normalize(mean=[0.5], std=[0.5]),         # center scale
        ])
        val_tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),])
    return train_tfms, val_tfms

def make_datasets(cfg: Config, train_tfms, val_tfms):
    # Compose absolute path to the train directory under the provided data_root
    train_dir = os.path.join(cfg.data_root, cfg.train_dirname)

    # Create the full dataset object (ImageFolder expects folder-per-class)
    full_ds = ImageFolder(root=train_dir, transform=train_tfms)

    # Verify classes in the dataset match the expected set 
    found_classes = full_ds.classes  # list of class names discovered by ImageFolder (sorted alphabetically)
    # Check we have exactly the 7 expected emotions
    assert set(found_classes) == set(EMOTIONS), \
        f"Expected classes {EMOTIONS}, but found {found_classes} in {train_dir}"

    # Build a label map we will use later for reports (idx -> class name)
    idx_to_class = {idx: cls_name for cls_name, idx in full_ds.class_to_idx.items()}

    # Collect labels for stratification (ImageFolder stores (path, label) in samples)
    all_labels = [lbl for _, lbl in full_ds.samples]

    # Compute stratified indices for train/val
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_split, random_state=cfg.seed)
    # splitter.split yields (train_idx, val_idx) based on labels
    train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))

    # Create "views" (subsets) of the dataset for train and val
    train_ds = Subset(full_ds, indices=train_idx)
    # validation uses val transforms, so we override subset's dataset transform
    val_base = ImageFolder(root=train_dir, transform=val_tfms)  # reload with val tfms
    val_ds = Subset(val_base, indices=val_idx)

    return train_ds, val_ds, idx_to_class, found_classes

def compute_class_weights_from_indices(full_samples: List[Tuple[str, int]],
                                       subset_indices: np.ndarray,
                                       num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a subset of the dataset.
    This will be used in CrossEntropyLoss(weight=...).

    Returns a tensor of shape [num_classes].
    """
    # Count occurrences of each class within the subset
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in subset_indices:
        _, y = full_samples[i]
        counts[y] += 1

    # Avoid division by zero; if any class has zero count (shouldn't happen with stratification),
    # we set a minimum of 1.
    counts = np.maximum(counts, 1)

    # Compute inverse frequency (heavier weight for rarer classes)
    inv = 1.0 / counts.astype(np.float32)
    # Normalizing so that mean weight = 1.0 (keeps loss scale reasonable)
    weights = inv * (num_classes / inv.sum())
    # Convert to torch tensor
    return torch.tensor(weights, dtype=torch.float32)

class SimpleCNN(nn.Module):
    """ A compact CNN suitable for 48x48 grayscale FER-2013.
    notes:
    - In: 1x48x48
    - 3 conv blocks -> adaptive pooling -> linear classifier (7 classes)
    - Dropout and BatchNorm for regularization & stabilized training """
    def __init__(self, num_classes: int = 7, in_ch: int = 1):
        super().__init__()

        # Block 1: conv -> bn -> relu -> pool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 48 -> 24
            nn.Dropout(0.10),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 24 -> 12
            nn.Dropout(0.15),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 12 -> 6
            nn.Dropout(0.20),
        )

        # Global pooling to remove spatial dims robustly
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # output: 128 x 1 x 1

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),)
    def forward(self, x):
        x = self.block1(x)  # feature extraction block 1
        x = self.block2(x)  # feature extraction block 2
        x = self.block3(x)  # feature extraction block 3
        x = self.gap(x)     # global average pool
        x = self.classifier(x)  # linear layers for classification
        return x
def build_model(cfg: Config, num_classes: int) -> nn.Module:
    """Create the selected model architecture.
    - "simple_cnn" -> SimpleCNN(1-channel input).
    - "resnet18"   -> torchvision.models.resnet18 with adjusted final layer.
                      If cfg.pretrained is True, attempts to load ImageNet weights
                      fall back to random init if not available)."""
    if cfg.model == "resnet18":
        # Try to create resnet18 with or without pretrained weights
        weights = None
        if cfg.pretrained:
            try:
                # The weights alias name varies by torchvision version; this should work on recent versions
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None  # fall back if weights enum unavailable
        model = models.resnet18(weights=weights)
        # Replaces the last fully-connected layer to output num_classes
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    else:
        # Using our small CNN (grayscale)
        return SimpleCNN(num_classes=num_classes, in_ch=1)

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, cfg: Config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # iterates loader with progress bar
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision context
        with autocast(enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Scaled backpropagation
        scaler.scale(loss).backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Stats
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(model, loader, device, criterion, cfg: Config):
    #Standard validation epoch  
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc

@torch.no_grad()
def predict_all(model, loader, device):
    """
    Run the model over a DataLoader and collect:
    - y_true: ground truth labels
    - y_pred: predicted integer labels
    - y_prob: softmax probabilities
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    for images, targets in tqdm(loader, desc="Eval ", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # class probabilities
        confs, preds = probs.max(dim=1)        # predicted labels

        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, path: str, extra: dict = None) -> None:
    """
    Saves a checkpoint containing model & optimizer state plus metadata.
    """
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_label_map(idx_to_class: dict, path: str) -> None:
    """Persist idx->class mapping as JSON for inference later."""
    with open(path, "w") as f:
        json.dump(idx_to_class, f, indent=2)


def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str) -> None:
    """
    Simple confusion matrix heatmap saved to disk.
    """
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # annotate cells
    thresh = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, 'd'),
                     ha="center", va="center",
                     color="white" if val > thresh * 0.6 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_image_for_inference(img_path: str, cfg: Config):
    """
    Loads a single image file and apply the correct validation transformsto prepare it for model inference.
    """
    from PIL import Image

    # Builds "val" transforms to be consistent with evaluation pipeline
    _, val_tfms = build_transforms(cfg)

    # Open with PIL
    img = Image.open(img_path).convert("RGB")  # open as RGB; tfms handle grayscale if needed
    # If simple_cnn, we will convert to grayscale in transforms as defined
    tensor = val_tfms(img).unsqueeze(0)  # add batch dimension
    return tensor


@torch.no_grad()
def predict_single_image(model, img_path: str, cfg: Config, device, idx_to_class: dict) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Run inference on a single image path and return:
      - predicted class label
      - list of (class_name, probability) pairs for all classes
    """
    x = load_image_for_inference(img_path, cfg).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    # pair up class names with probabilities
    named_probs = [(idx_to_class[i], float(p)) for i, p in enumerate(probs)]
    named_probs.sort(key=lambda t: t[1], reverse=True)
    return pred_label, named_probs


# Orchestration – Main flow
def main():
    #   Parse CLI args into Config  
    parser = argparse.ArgumentParser(description="FER-2013 Trainer")
    parser.add_argument("--data-root", type=str, default="./archive", help="Root folder that contains 'train' subfolder.")
    parser.add_argument("--train-dirname", type=str, default="train", help="Name of the training subfolder under data-root.")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet18"], help="Model architecture.")
    parser.add_argument("--img-size", type=int, default=48, help="Input image size (48 for simple_cnn, e.g. 224 for resnet18).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train set used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--pretrained", action="store_true", help="Try to load pretrained ImageNet weights (ResNet-18 only).")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max-norm for gradient clipping (0 to disable).")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-imbalance loss weighting.")
    parser.add_argument("--save-dir", type=str, default="./outputs", help="Directory to save checkpoints and reports.")
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        train_dirname=args.train_dirname,
        model=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        use_amp=not args.no_amp,
        patience=args.patience,
        grad_clip=args.grad_clip,
        no_class_weights=args.no_class_weights,
        save_dir=args.save_dir,
    )

    #   Setup output directory and reproducibility  
    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed)
    #   Device selection  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")
    train_tfms, val_tfms = build_transforms(cfg)
    train_dir = os.path.join(cfg.data_root, cfg.train_dirname)
    full_for_counts = ImageFolder(root=train_dir, transform=transforms.ToTensor())
    train_ds, val_ds, idx_to_class, found_classes = make_datasets(cfg, train_tfms, val_tfms)
    pin = True if device.type == "cuda" else False
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    num_classes = len(found_classes)
    model = build_model(cfg, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    if cfg.no_class_weights:
        class_weights = None
        print("[Info] Class-imbalance loss weighting: OFF")
    else:
        train_indices = train_ds.indices if hasattr(train_ds, "indices") else list(range(len(train_ds)))
        class_weights = compute_class_weights_from_indices(
            full_samples=full_for_counts.samples,
            subset_indices=np.array(train_indices),
            num_classes=num_classes
        ).to(device)
        print(f"[Info] Class weights: {class_weights.detach().cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler(enabled=cfg.use_amp)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    labelmap_path = os.path.join(cfg.save_dir, "label_map.json")
    save_label_map(idx_to_class, labelmap_path)

    # Training loop 
    history = []  
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        # for one epoch
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg)
        # validate
        val_loss, val_acc = validate_one_epoch(model, val_loader, device, criterion, cfg)
        # LR scheduler on validation loss
        scheduler.step(val_loss)
        # log
        print(f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc})

        # checkpoint last
        save_checkpoint(model, optimizer, epoch,
                        path=os.path.join(cfg.save_dir, "last_model.pth"),
                        extra={"config": vars(cfg), "label_map": idx_to_class})

        # early stopping check
        if val_loss < best_val_loss - 1e-4:  
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch,
                            path=os.path.join(cfg.save_dir, "best_model.pth"),
                            extra={"config": vars(cfg), "label_map": idx_to_class})
            print(f"[Checkpoint] New best model saved (epoch {epoch})")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStop] No improvement for {epochs_no_improve} epoch(s) (patience {cfg.patience})")
            if epochs_no_improve >= cfg.patience:
                print("[EarlyStop] Stopping training.")
                break
    print("\n[Evaluation] Computing predictions on validation set...")
    y_true, y_pred, _ = predict_all(model, val_loader, device=device)
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nClassification report:\n")
    print(report)
    report_path = os.path.join(cfg.save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[Saved] {report_path}")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_path = os.path.join(cfg.save_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, class_names=target_names, out_path=cm_path)
    print(f"[Saved] {cm_path}")
    hist_path = os.path.join(cfg.save_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] {hist_path}")
    print(f"\n[Done] Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"       Checkpoints: best_model.pth / last_model.pth in {cfg.save_dir}")
    print(f"       Label map: {labelmap_path}")

if __name__ == "__main__":
    main()
