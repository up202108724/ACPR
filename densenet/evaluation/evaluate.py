import torch
from train.validate import validate_model

def test_model(model, test_loader, criterion, device):
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
    metrics = validate_model(model, test_loader, criterion, device)

    print("🧪 Test Results")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k.capitalize()}: {v:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])