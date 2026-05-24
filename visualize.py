import torch
import matplotlib.pyplot as plt

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
    batch_size=8,
    shuffle=True

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
# GET SAMPLE BATCH
# ---------------------------------------------------

images, labels = next(iter(test_loader))

images = images.to(device)
labels = labels.to(device)


# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

with torch.no_grad():

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)


# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

axes = axes.flatten()

for i in range(8):

    # Convert image format:
    # [C, H, W] -> [H, W, C]

    image = images[i].cpu().permute(1, 2, 0)

    axes[i].imshow(image)

    axes[i].set_title(

        f"Pred: {predicted[i].item()}\n"
        f"True: {labels[i].item()}"

    )

    axes[i].axis("off")


plt.tight_layout()

plt.show()