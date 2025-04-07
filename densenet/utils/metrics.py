from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import torch

def compute_metrics(outputs, labels):
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    return {
        "accuracy": 100 * np.mean(preds == labels),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds)
    }