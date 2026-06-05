"""
Configuration file for Knowledge Distillation Ensemble project.
All hyperparameters and path settings are centralized here.
"""

import torch

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
NUM_EPOCHS    = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4

# Learning rate scheduler: divide by 10 at 50% and 75% of total epochs
LR_MILESTONES = [0.5, 0.75]   # fractions of NUM_EPOCHS
LR_GAMMA      = 0.1

# ──────────────────────────────────────────────
# KD Loss Weights
# ──────────────────────────────────────────────
# Response-based (Logits KD) — Eq. 3
LOGITS_ALPHA = 1.0   # student loss weight
LOGITS_BETA  = 0.5   # distillation loss weight

# Relation-based (RKD) — Eq. 6
RKD_DISTANCE_WEIGHT = 1.0
RKD_ANGLE_WEIGHT    = 2.0

# ──────────────────────────────────────────────
# Teacher-Dataset Assignment
# ──────────────────────────────────────────────
# Each student is paired with its own teacher and dataset:
#
#   Student-Logits  ←  Teacher-200k  on  Dataset-200k
#   Student-AT      ←  Teacher-140k  on  Dataset-140k
#   Student-RKD     ←  Teacher-190k  on  Dataset-190k
#
# This ensures each student learns complementary knowledge
# from a different source, making the ensemble more diverse.

TEACHER_PATHS = {
    '200k': '/kaggle/input/datasets/sarah20079/teacher-model-best-200k/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth',   # used by Student-Logits
    '140k': '/kaggle/input/models/sara24h/teacher_model_best/pytorch/default/1/teacher_model_best.pth',   # used by Student-AT
    '190k': '/kaggle/input/datasets/sara24h/kdfs-190k-transfer-learning-data/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth',   # used by Student-RKD
}

DATASET_PATHS = {
    '200k': {
        'train_csv': '/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv',
        'val_csv':   '/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv',
        'test_csv':  '/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
        'root_dir':  '/kaggle/input/undersampled-200k/balanced_unique_200k_dataset',
    },
    '140k': {
        'train_csv': '/kaggle/input/140k-real-and-fake-faces/train.csv',
        'val_csv':   '/kaggle/input/140k-real-and-fake-faces/valid.csv',
        'test_csv':  '/kaggle/input/140k-real-and-fake-faces/test.csv',
        'root_dir':  '/kaggle/input/140k-real-and-fake-faces',
    },
    '190k': {
        'root_dir':  '/kaggle/input/deepfake-and-real-images/Dataset',
    },
}

SAVE_DIR = './kd_checkpoints'

# Which dataset's test split to use for final evaluation
# Can be '200k', '140k', '190k' — or evaluate on all separately
TEST_DATASET = '200k'
