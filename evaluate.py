import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_model import TrafficSignCNN


# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ---------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor()

])


# ---------------------------------------------------
# TEST DATASET
# ---------------------------------------------------

test_dataset = datasets.GTSRB(

    root="./data",
    split="test",
    transform=transform,
    download=True

)


# ---------------------------------------------------
# DATALOADER
# ---------------------------------------------------

test_loader = DataLoader(

    test_dataset,
    batch_size=64,
    shuffle=False

)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = TrafficSignCNN().to(device)

model.load_state_dict(
    torch.load("traffic_sign_cnn.pth")
)

model.eval()


# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


# ---------------------------------------------------
# FINAL ACCURACY
# ---------------------------------------------------

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")