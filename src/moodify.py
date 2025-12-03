import pathlib
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

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
    ''' 
    Normalizes, reshapes, and augments the data (applies transformations to tensor, guaranteeing the images are all in grayscale [0, 1])
    Args: None
    Returns:
        cnn_train_tf, cnn_test_tf: Transformations for CNN model.
        resnet_train_tf, resnet_test_tf: Transformations for ResNet model.
    '''
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
        transforms.Normalize([GRAY_MEAN], [GRAY_STD])
    ])

    cnn_test_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([GRAY_MEAN], [GRAY_STD])
    ])

    resnet_train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    resnet_test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf
    
# =============
# DATALOADERS
# =============
def create_dataloaders_imagefolder(data_dir, cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf):
    '''
    Creates DataLoaders for CNN and ResNet models using ImageFolder.
    Args:
        data_dir (str): Path to the dataset directory.
        cnn_train_tf, cnn_test_tf: Transformations for CNN model.
        resnet_train_tf, resnet_test_tf: Transformations for ResNet model.
    Returns:
        Tuple of DataLoaders: (cnn_train_loader, cnn_test_loader, resnet_train_loader, resnet_test_loader)
    '''

    # CNN datasets
    cnn_train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=cnn_train_tf)
    cnn_test_ds  = ImageFolder(os.path.join(data_dir, "test"),  transform=cnn_test_tf)

    # ResNet datasets
    resnet_train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=resnet_train_tf)
    resnet_test_ds  = ImageFolder(os.path.join(data_dir, "test"),  transform=resnet_test_tf)

    # DataLoaders
    cnn_train_loader   = DataLoader(cnn_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cnn_test_loader    = DataLoader(cnn_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    resnet_train_loader = DataLoader(resnet_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    resnet_test_loader  = DataLoader(resnet_test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # use class names from ImageFolder (is alphabetically sorted)
    classes = cnn_train_ds.classes
    
    return (
        cnn_train_loader, cnn_test_loader,
        resnet_train_loader, resnet_test_loader,
        classes
    )

    
# =============
# CNN
# =============
class CustomCNN(nn.Module):
    '''
    A simple Convolutional Neural Network for facial emotion recognition.
    3 convolutional layers followed by 2 fully connected layers.
    48x48 grayscale input images.
    7 output classes.
    '''
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (1 → 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                # 48 → 24
            nn.Conv2d(32, 64, 3, padding=1), # 32 → 64
            nn.ReLU(),
            nn.MaxPool2d(2),                # 24 → 12
            nn.Conv2d(64, 128, 3, padding=1), # 64 → 128
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 12 → 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============
# RESNET18
# =============
def get_resnet18(num_classes=7):
    '''
    Loads a pre-trained ResNet18 model and modifies the final layer for emotion classification.
    Args:
        num_classes (int): Number of output classes.
    Returns:
        model (nn.Module): Modified ResNet18 model.
    '''
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False  # freeze backbone
    
    # Unfreezing the last block (which is layer4 + fc) for fine tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Gives head more capacity (including a Dropout)
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
def train_model(model, train_loader, test_loader, epochs=5):
    '''
    Trains the given model using the provided DataLoaders.
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
    Returns:
        model (nn.Module): The trained model.
    '''
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Choose params to optimize
    if hasattr(model, "fc"):   # ResNet18 path (since backbone is frozen, no need for other params)
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-4) # smaller lr since we unfreeze last 2 layers
    else:                      # CustomCNN or other models
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Acc={acc:.4f}")

    evaluate(model, test_loader)
    return model

# =============
# EVAL LOOP
# =============
def evaluate(model, loader):
    '''
    Evaluates the given model on the provided DataLoader.
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for evaluation data.
    Returns: None
    '''
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
    print("Test Accuracy:", correct / len(loader.dataset))
    
# =============
# INFERENCE
# =============
def predict_image(model, path, transform, classes):
    '''
    Predicts the emotion class for a given image using the specified model and transformation.
    Args:
        model (nn.Module): The model to use for prediction.
        path (str): Path to the image file.
        transform (callable): Transformation to apply to the image.
        classes (list): List of class names.
    Returns:
        str: Predicted emotion class.
    '''
    img = Image.open(path).convert("L")  # FER images are grayscale
    img = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(img).argmax(1).item()

    return classes[pred]

# =============
# CLI INTERFACE
# =============
def cli_interface(INFERENCE_DIR, cnn_model, resnet_model, cnn_tf, resnet_tf, classes):
    '''
    Command-line interface for user to select image and model for emotion prediction.
    Args:
        INFERENCE_DIR (str): Directory containing images for inference.
        cnn_model (nn.Module): Trained CNN model.
        resnet_model (nn.Module): Trained ResNet model.
        cnn_tf (callable): Transformation for CNN model.
        resnet_tf (callable): Transformation for ResNet model.
        classes (list): List of class names.
    Returns: None
    '''
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
    '''
    Main function to run the MOODIFY facial emotion recognition system.
    Steps:
    1. Load dataset.
    2. Apply transformations.
    3. Create DataLoaders.
    4. Initialize and train models (Custom CNN and ResNet18).
    5. Launch CLI for inference.
    Returns: None
    '''
    DATA_DIR = load_dataset()
    INFERENCE_DIR = "./inference_images" # drop images here for inference
    
    print("Welcome to MOODIFY - Facial Emotion Recognition System")
    # Transfomrations
    cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf = transformations()
    
    # create dataloaders
    (
        cnn_train_loader, cnn_test_loader,
        resnet_train_loader, resnet_test_loader,
        classes
    ) = create_dataloaders_imagefolder(DATA_DIR, cnn_train_tf, cnn_test_tf, resnet_train_tf, resnet_test_tf)
    
    # Sanity checks
    x_cnn, y_cnn = next(iter(cnn_train_loader))
    print("CNN batch:", x_cnn.shape, x_cnn.min().item(), x_cnn.max().item())

    x_res, y_res = next(iter(resnet_train_loader))
    print("ResNet batch:", x_res.shape, x_res.min().item(), x_res.max().item())

    # initialize models
    num_classes = len(classes)
    cnn_model = CustomCNN(num_classes=num_classes)
    resnet_model = get_resnet18(num_classes=num_classes)
    
    # train both
    print("\n   Training Custom CNN...")
    cnn_model = train_model(cnn_model, cnn_train_loader, cnn_test_loader, epochs=10) # Reusing same name?

    print("\n   Training ResNet18...")
    resnet_model = train_model(resnet_model, resnet_train_loader, resnet_test_loader, epochs=10) # Reusing same name?
    
    # Saving Models
    torch.save(cnn_model.state_dict(), "cnn_frozen.pth")
    torch.save(resnet_model.state_dict(), "resnet_frozen.pth")
    
    # cli interface
    cli_interface(INFERENCE_DIR, cnn_model, resnet_model, cnn_test_tf, resnet_test_tf, classes)
    
if __name__ == "__main__":
    main()