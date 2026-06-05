"""
evaluation.py
=============
Evaluation utilities for single models and soft-voting ensembles.

  evaluate()             → metrics for one model
  evaluate_ensemble()    → metrics for ensemble of student models  (Eq. 9)
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate a single model on *dataloader*.

    Returns
    -------
    dict  with keys: accuracy, precision, recall, f1, auc
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob)
                     if len(np.unique(y_true)) > 1 else 0.0,
    }


@torch.no_grad()
def evaluate_ensemble(student_models, dataloader, device):
    """
    Soft-voting ensemble evaluation  (Eq. 9).

    ŷ = argmax_i  Σ_j  z_ij

    For binary classification: average the logits from all student models
    → sigmoid → threshold 0.5.
    """
    for m in student_models:
        m.eval()

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in dataloader:
        images = images.to(device)

        # sum logits from all students, then divide
        avg_logits = sum(m(images) for m in student_models) / len(student_models)
        probs = torch.sigmoid(avg_logits).squeeze()
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob)
                     if len(np.unique(y_true)) > 1 else 0.0,
    }
 
