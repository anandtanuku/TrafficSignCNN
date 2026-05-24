import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_model import TrafficSignCNN

transform = transforms.Compose([

    transforms.Resize((32,32)),

    transforms.ToTensor()

])

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

train_dataset = datasets.GTSRB(
    root = './data',
    split = 'train',
    transform = transform,
    download = True
)

test_dataset = datasets.GTSRB(
    root = './data',
    split = 'test',
    transform = transform,
    download = True
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

model = TrafficSignCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

# training loop

num_epochs = 5

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    

print(images.shape)


# save model

torch.save(model.state_dict(), "traffic_sign_cnn.pth")

print("Model saved successfully.")
