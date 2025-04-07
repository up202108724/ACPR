from tqdm import tqdm
from utils.metrics import compute_metrics
from train.validate import validate_model
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, writer, num_epochs=10):
    best_acc = 0.0
    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        val_metrics = validate_model(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")