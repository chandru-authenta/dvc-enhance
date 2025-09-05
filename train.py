import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import yaml, json

# Load hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS = params["epochs"]
BATCH_SIZE = params["batch_size"]
LR = params["learning_rate"]
IMG_SIZE = params["img_size"]

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Simple CNN model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # fake vs real

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
history = {"loss": [], "accuracy": []}

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    history["loss"].append(epoch_loss)
    history["accuracy"].append(epoch_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "model.pth")

# Save metrics for DVC
with open("metrics.json", "w") as f:
    json.dump({"loss": history["loss"][-1], "accuracy": history["accuracy"][-1]}, f)
