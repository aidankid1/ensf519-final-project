import pathlib
from pathlib import Path
import time
import kagglehub
import torch.nn as nn
import torch.optim as optim
import os
import torch
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# =============
# GLOBAL CONFIG
# =============
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
RESNET_MODE = "full"   # "fc" | "layer4" | "full"
EPOCHS_CNN = 100
EPOCHS_RESNET = 100

# speed up convs on fixed-size inputs
torch.backends.cudnn.benchmark = True

# num_workers for DataLoader (use most cores but keep 1 free)
CPU_CORES = os.cpu_count() or 4
NUM_WORKERS = max(1, min(8, CPU_CORES - 1))
PIN_MEMORY = DEVICE.type == "cuda"

# Folder for saved models
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)  # create if not exists

# Canonical filenames (need to change)
CNN_WEIGHTS_PATH = SAVE_DIR / "cnn_latest.pth"
RESNET_WEIGHTS_PATH = SAVE_DIR / f"resnet_latest_{RESNET_MODE}.pth"

# =============
# LOAD DATASET
# =============
def load_dataset() -> str:
    """
    Downloads the FER-2013 facial expression dataset from Kaggle and returns the path to the dataset files.
    """
    print("Downloading FER-2013 dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("msambare/fer2013")
    except Exception as e:
        print("An error occurred while downloading the dataset:", e)
        raise
    print("Dataset downloaded successfully. Path to dataset files:", path)
    return path


# =============
# TRANSFORMS
# =============
def transformations():
    """
    Normalizes, reshapes, and augments the data.
    Returns:
        cnn_train_tf, cnn_test_tf: Transformations for CNN model.
        resnet_train_tf, resnet_test_tf: Transformations for ResNet model.
    """
    GRAY_MEAN = 0.5
    GRAY_STD = 0.5
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    cnn_train_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([GRAY_MEAN], [GRAY_STD]),
    ])

    cnn_test_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([GRAY_MEAN], [GRAY_STD]),
    ])

    resnet_train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    resnet_test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf


# =============
# DATALOADERS
# =============
def create_dataloaders_imagefolder(data_dir, cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf):
    """
    Creates DataLoaders for CNN and ResNet models using ImageFolder.
    """
    # CNN datasets
    cnn_train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=cnn_train_tf)
    cnn_test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=cnn_test_tf)

    # ResNet datasets
    resnet_train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=resnet_train_tf)
    resnet_test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=resnet_test_tf)

    # DataLoaders (GPU‑friendly settings)
    cnn_train_loader = DataLoader(
        cnn_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
    )
    cnn_test_loader = DataLoader(
        cnn_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
    )

    resnet_train_loader = DataLoader(
        resnet_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
    )
    resnet_test_loader = DataLoader(
        resnet_test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # use class names from ImageFolder (alphabetically sorted)
    classes = cnn_train_ds.classes

    return (
        cnn_train_loader, cnn_test_loader,
        resnet_train_loader, resnet_test_loader,
        classes,
    )

class OldCustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============
# CNN
# =============
class CustomCNN(nn.Module):
    """
    A Convolutional Neural Network for facial emotion recognition.
    3 conv blocks with BatchNorm + larger FC head.
    48x48 grayscale input images.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 48 -> 24

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 24 -> 12

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),       # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============
# RESNET18
# =============
def get_resnet18(num_classes=7, mode="layer4"):
    """
    Loads a pre-trained ResNet18 and configures which parts to fine-tune.
    mode: "fc" | "layer4" | "full"
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 1) Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    # 2) Decide what to unfreeze based on mode
    if mode == "fc":
        # only final classifier gets trained
        pass
    elif mode == "layer4":
        # unfreeze last conv block
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode == "full":
        # unfreeze ALL layers
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown ResNet mode: {mode}")

    # Replace head with bigger classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    return model


# =============
# TRAINING LOOP
# =============
def train_model(model, train_loader, test_loader, epochs=5, mode="cnn"):
    """
    Trains the given model using the provided DataLoaders.
    mode:
        - "cnn"          -> CustomCNN, all params, lr=1e-3
        - "resnet_fc"    -> only fc, lr=1e-3
        - "resnet_layer4"-> layer4+fc, lr=1e-4
        - "resnet_full"  -> all params, lr=1e-5
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # choose params + LR
    if mode == "cnn":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif mode == "resnet_fc":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
    elif mode == "resnet_layer4":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
        )
    elif mode == "resnet_full":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5,
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()
        total_loss = 0.0
        correct = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if i == 0:
                print(f"[DEBUG] First batch device: {imgs.device}, model on: {next(model.parameters()).device}")

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        epoch_time = time.time() - start_time
        acc = correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={avg_loss:.4f}, Train Acc={acc:.4f}, "
            f"Time={epoch_time:.1f}s"
        )

    evaluate(model, test_loader)
    return model


# =============
# EVAL LOOP
# =============
def evaluate(model, loader):
    """Evaluates the given model on the provided DataLoader."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
    print("Test Accuracy:", correct / len(loader.dataset))


# =============
# INFERENCE
# =============
def predict_image(model, path, transform, classes):
    """
    Predicts the emotion class for a given image using the specified model and transformation.
    """
    img = Image.open(path).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(img).argmax(1).item()

    return classes[pred]

def train_and_save_models(
    cnn_train_loader,
    cnn_test_loader,
    resnet_train_loader,
    resnet_test_loader,
    classes,
):
    num_classes = len(classes)

    print("\n   Training Custom CNN...")
    cnn_model = CustomCNN(num_classes=num_classes)
    cnn_model = train_model(
        cnn_model,
        cnn_train_loader,
        cnn_test_loader,
        epochs=EPOCHS_CNN,
        mode="cnn",
    )
    torch.save(cnn_model.state_dict(), CNN_WEIGHTS_PATH)
    print(f"Saved CNN weights to {CNN_WEIGHTS_PATH}")

    if RESNET_MODE == "fc":
        resnet_mode_name = "resnet_fc"
    elif RESNET_MODE == "layer4":
        resnet_mode_name = "resnet_layer4"
    else:
        resnet_mode_name = "resnet_full"

    print("\n   Training ResNet18...")
    resnet_model = get_resnet18(num_classes=num_classes, mode=RESNET_MODE)
    resnet_model = train_model(
        resnet_model,
        resnet_train_loader,
        resnet_test_loader,
        epochs=EPOCHS_RESNET,
        mode=resnet_mode_name,
    )
    torch.save(resnet_model.state_dict(), RESNET_WEIGHTS_PATH)
    print(f"Saved ResNet weights to {RESNET_WEIGHTS_PATH}")

    cnn_model.eval()
    resnet_model.eval()
    return cnn_model, resnet_model

def load_models_from_disk(num_classes: int):
    # 1) find all run folders inside saved_models/
    if not SAVE_DIR.exists():
        print("\nNo 'saved_models' folder found yet. Please train first.\n")
        return None, None

    run_dirs = sorted([d for d in SAVE_DIR.iterdir() if d.is_dir()])
    if not run_dirs:
        print("\nNo saved runs found inside 'saved_models/'. Please train first.\n")
        return None, None

    # 2) let user choose which run (date) to load
    print("\n=== Saved Model Runs ===")
    for i, d in enumerate(run_dirs):
        print(f"[{i}] {d.name}")
    print("[b] Back to main menu")

    while True:
        choice = input("Select a run index to load: ").strip().lower()
        if choice == "b":
            return None, None
        if not choice.isdigit():
            print("Please enter a valid index or 'b' to go back.")
            continue

        idx = int(choice)
        if idx < 0 or idx >= len(run_dirs):
            print("Index out of range, try again.")
            continue

        run_dir = run_dirs[idx]
        break

    # 3) inside the chosen run folder, find CNN + ResNet .pth files
    cnn_files = list(run_dir.glob("cnn_*.pth"))
    resnet_files = list(run_dir.glob("resnet_*.pth"))

    if not cnn_files or not resnet_files:
        print(f"\nSelected run '{run_dir.name}' does not contain both cnn_*.pth and resnet_*.pth.")
        print("Files found:")
        print("  CNN:", [p.name for p in cnn_files] or "None")
        print("  ResNet:", [p.name for p in resnet_files] or "None")
        return None, None

    # take the first match of each (you only have one of each per folder)
    cnn_path = cnn_files[0]
    resnet_path = resnet_files[0]

    print(f"\nLoading CNN from {cnn_path} ...")
    cnn_model = CustomCNN(num_classes=num_classes).to(DEVICE)
    cnn_state = torch.load(cnn_path, map_location=DEVICE)
    cnn_model.load_state_dict(cnn_state)
    cnn_model.eval()

    print(f"Loading ResNet from {resnet_path} ...")
    resnet_model = get_resnet18(num_classes=num_classes, mode=RESNET_MODE).to(DEVICE)
    resnet_state = torch.load(resnet_path, map_location=DEVICE)
    resnet_model.load_state_dict(resnet_state)
    resnet_model.eval()

    print("\nLoaded models from run:", run_dir.name)
    return cnn_model, resnet_model


# =============
# CLI INTERFACE
# =============
def cli_interface(INFERENCE_DIR, cnn_model, resnet_model, cnn_tf, resnet_tf, classes):
    """Command-line interface for inference."""
    while True:
        print("\n=== EMOTION CLASSIFIER CLI ===")
        print(f"Place images in: {INFERENCE_DIR}\n")

        images = list(pathlib.Path(INFERENCE_DIR).glob("*"))
        if not images:
            print("No images found.")
            return

        for i, img in enumerate(images):
            print(f"[{i}] {img.name}")
        print("[q] Quit")

        choice_idx = input("\nSelect image index (or 'q' to quit): ")
        if choice_idx.lower() == "q":
            break

        idx = int(choice_idx)
        img_path = str(images[idx])

        print("\nChoose model:")
        print("[0] Custom CNN")
        print("[1] ResNet18")

        model_choice = input("Your choice: ")
        if model_choice == "0":
            pred = predict_image(cnn_model, img_path, cnn_tf, classes)
        elif model_choice == "1":
            pred = predict_image(resnet_model, img_path, resnet_tf, classes)
        else:
            print("Invalid choice, returning to menu.")
            continue

        print(f"\nPredicted Emotion: {pred}")


def main():
    """
    Main function to run the MOODIFY facial emotion recognition system.
    """
    DATA_DIR = load_dataset()
    INFERENCE_DIR = "./inference_images"  # drop images here for inference

    print("Welcome to MOODIFY - Facial Emotion Recognition System")
    print("Using device:", DEVICE)
    # print("DataLoader workers:", NUM_WORKERS, "pin_memory:", PIN_MEMORY)

    # Transformations
    cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf = transformations()

    # Dataloaders
    (
        cnn_train_loader, cnn_test_loader,
        resnet_train_loader, resnet_test_loader,
        classes
    ) = create_dataloaders_imagefolder(
        DATA_DIR, cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf
    )

    num_classes = len(classes)

    # Optional sanity checks (keep or comment out)
    # x_cnn, y_cnn = next(iter(cnn_train_loader))
    # print("CNN batch:", x_cnn.shape, x_cnn.min().item(), x_cnn.max().item())
    # x_res, y_res = next(iter(resnet_train_loader))
    # print("ResNet batch:", x_res.shape, x_res.min().item(), x_res.max().item())

    # ==========
    # MAIN MENU
    # ==========
    while True:
        print("\n=== MOODIFY MAIN MENU ===")
        print("[1] Train new models (CNN + ResNet) and save them")
        print("[2] Load previously saved models from 'saved_models/'")
        print("[q] Quit")

        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            cnn_model, resnet_model = train_and_save_models(
                cnn_train_loader,
                cnn_test_loader,
                resnet_train_loader,
                resnet_test_loader,
                classes,
            )
            break

        elif choice == "2":
            cnn_model, resnet_model = load_models_from_disk(num_classes)
            if cnn_model is None or resnet_model is None:
                # loop again so user can choose to train instead
                continue
            break

        elif choice == "q":
            print("Exiting MOODIFY. Goodbye!")
            return

        else:
            print("Invalid choice, please select 1, 2, or q.")

    # After we have models, run CLI for image prediction
    cli_interface(INFERENCE_DIR, cnn_model, resnet_model, cnn_test_tf, resnet_test_tf, classes)


if __name__ == "__main__":
    main()
