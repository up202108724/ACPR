import torch
from utils.metrics import compute_metrics

def validate_model(model, dataloader, criterion, device):
    model.eval()
    all_outputs, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            all_outputs.append(outputs)
            all_labels.append(labels)
            total_loss += loss.item()

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics