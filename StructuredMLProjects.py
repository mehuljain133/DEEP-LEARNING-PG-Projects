# Unit-VI Structuring Machine Learning Projects: Orthogonalization, evaluation metrics, train/dev/test distributions, size of the dev and test sets, cleaning up incorrectly labeled data, bias and variance with mismatched data distributions, transfer learning, multi-task learning.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt

# Seed for reproducibility
torch.manual_seed(42)

# 1. Data loading and splitting
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Simulate label noise: flip 10% of labels
def corrupt_labels(dataset, corruption_rate=0.1):
    corrupted_targets = dataset.targets.clone()
    n = int(len(corrupted_targets) * corruption_rate)
    indices = torch.randperm(len(corrupted_targets))[:n]
    corrupted_targets[indices] = torch.randint(0, 10, (n,))
    dataset.targets = corrupted_targets

corrupt_labels(dataset)

# Train/dev/test split
train_size = int(0.7 * len(dataset))
dev_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - dev_size
train_set, dev_set, test_set = random_split(dataset, [train_size, dev_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

# 2. Simple CNN model (multi-task: digit + even/odd)
class MultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 24 * 24, 10)       # Digit classification
        self.binary_head = nn.Linear(32 * 24 * 24, 1)       # Even/Odd binary

    def forward(self, x):
        x = self.features(x)
        class_out = self.classifier(x)
        binary_out = torch.sigmoid(self.binary_head(x))
        return class_out, binary_out

# 3. Training logic
def train(model, loader, epochs=3):
    model.train()
    criterion_class = nn.CrossEntropyLoss()
    criterion_bin = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            binary_labels = (labels % 2 == 0).float().unsqueeze(1)
            images, labels, binary_labels = images.to(device), labels.to(device), binary_labels.to(device)

            class_out, binary_out = model(images)
            loss_class = criterion_class(class_out, labels)
            loss_binary = criterion_bin(binary_out, binary_labels)
            loss = loss_class + loss_binary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# 4. Evaluation metrics
def evaluate(model, loader, label="Test"):
    model.eval()
    all_preds, all_true = [], []
    all_binary_preds, all_binary_true = [], []

    with torch.no_grad():
        for images, labels in loader:
            binary_labels = (labels % 2 == 0).int()
            images = images.to(device)
            class_out, binary_out = model(images)
            preds = torch.argmax(class_out, dim=1).cpu()
            binary_preds = (binary_out > 0.5).int().cpu()
            all_preds.extend(preds)
            all_true.extend(labels)
            all_binary_preds.extend(binary_preds)
            all_binary_true.extend(binary_labels)

    print(f"\n📊 Evaluation on {label} Set")
    print("Digit Accuracy:", accuracy_score(all_true, all_preds))
    print("Even/Odd Precision:", precision_score(all_binary_true, all_binary_preds))
    print("Even/Odd Recall:", recall_score(all_binary_true, all_binary_preds))
    print("Even/Odd F1:", f1_score(all_binary_true, all_binary_preds))

# 5. Transfer learning (load pretrained CNN and fine-tune classifier)
def transfer_learning_demo():
    print("\n--- Transfer Learning Example ---")
    pretrained = models.resnet18(weights='IMAGENET1K_V1')
    pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    pretrained.fc = nn.Linear(pretrained.fc.in_features, 10)
    model = pretrained.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        break
    print("Transfer Learning Initial Batch Loss:", loss.item())

# Setup and run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train and evaluate multi-task model
model = MultiTaskNet().to(device)
train(model, train_loader)
evaluate(model, dev_loader, label="Dev")
evaluate(model, test_loader, label="Test")

# Transfer learning demo
transfer_learning_demo()
