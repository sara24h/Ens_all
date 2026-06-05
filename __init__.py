"""
kd-ensemble
===========

Ensemble Learning of Lightweight Deep Learning Models Using Knowledge
Distillation for Deepfake Image Classification.

Based on:
    Kang & Gwak, "Ensemble Learning of Lightweight Deep Learning Models
    Using Knowledge Distillation for Image Classification",
    Mathematics 2020.

Modules
-------
models       ResNet-50 with intermediate feature extraction
losses       KD loss functions (Logits, AT, RKD)
trainers     Training loops (single-teacher & multi-teacher)
evaluation   Single-model & ensemble evaluation (Soft Voting)
pipeline     End-to-end pipeline orchestration
predictor    Inference helper (EnsemblePredictor)
config       Hyper-parameters & path configuration
main         Entry point
"""

__version__ = "1.0.0"
