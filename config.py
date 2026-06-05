"""
config.py
=========
Central place for hyper-parameters and path configurations.

Edit the values below to match your environment before running `main.py`.
"""

import torch

# ── Device ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Training hyper-parameters ───────────────────────────────────────────
NUM_EPOCHS  = 50
LR          = 0.01
BATCH_SIZE  = 32

# Logits KD  (Eq. 3)
ALPHA = 1.0      # weight for student loss
BETA  = 0.5      # weight for logits distillation loss

# RKD  (Eq. 6)
RKD_DISTANCE_WEIGHT = 1.0
RKD_ANGLE_WEIGHT    = 2.0

# ── Paths — UPDATE THESE ────────────────────────────────────────────────
SAVE_DIR = "./kd_checkpoints"

TEACHER_PATHS = {
    "200k": "/path/to/teacher_200k_best.pth",
    "140k": "/path/to/teacher_140k_best.pth",
    "190k": "/path/to/teacher_190k_best.pth",
}

DATASET_PATHS = {
    "200k": {
        "train_csv":  "/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv",
        "val_csv":    "/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv",
        "test_csv":   "/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv",
        "root_dir":   "/kaggle/input/undersampled-200k/balanced_unique_200k_dataset",
    },
    "140k": {
        "train_csv":  "/kaggle/input/140k-real-and-fake-faces/train.csv",
        "valid_csv":  "/kaggle/input/140k-real-and-fake-faces/valid.csv",
        "test_csv":   "/kaggle/input/140k-real-and-fake-faces/test.csv",
        "root_dir":   "/kaggle/input/140k-real-and-fake-faces",
    },
    "190k": {
        "root_dir":   "/kaggle/input/deepfake-and-real-images/Dataset",
    },
}

# ── Pipeline options ────────────────────────────────────────────────────
MULTI_TEACHER  = True          # average distillation over all 3 teachers
TRAIN_DATASET  = "combined"    # '200k' | '140k' | '190k' | 'combined'
