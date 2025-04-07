import torch
from models.densenet import DenseNet3
from data.dataloader import get_dataloaders
from train.train import train_model
from evaluation.evaluate import test_model
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=f"runs/deepfake_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")


# Data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])  # your transforms
print("Loading data...")
train_loader, val_loader, test_loader = get_dataloaders("images/", transform)
print(" Data ready!")
# Train
print("Initializing model...")

# Model
model = DenseNet3(depth=40, num_classes=2, growth_rate=12, dropRate=0.1).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

print("🚦 Beginning training process...")
train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, num_epochs=20)

print("🧪 Evaluating on test set...")
# Test
test_model(model, test_loader, criterion, device)

print("✅ Done!")