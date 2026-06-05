"""
Main entry point — run the full KD ensemble pipeline.

Usage:
    python main.py

Make sure to update config.py with your paths before running.
"""

import torch
from config import *
from dataset import Dataset_selector
from pipeline import run_pipeline


def main():
    print("=" * 70)
    print("  Knowledge Distillation Ensemble for Deepfake Detection")
    print("  Based on: Kang & Gwak, Mathematics 2020")
    print("=" * 70)

    # ── Create datasets ──
    datasets = {}

    try:
        datasets['200k'] = Dataset_selector(
            dataset_mode='200k',
            realfake200k_train_csv=DATASET_PATHS['200k']['train_csv'],
            realfake200k_val_csv=DATASET_PATHS['200k']['val_csv'],
            realfake200k_test_csv=DATASET_PATHS['200k']['test_csv'],
            realfake200k_root_dir=DATASET_PATHS['200k']['root_dir'],
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"  Skipping 200k dataset: {e}")

    try:
        datasets['140k'] = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv=DATASET_PATHS['140k']['train_csv'],
            realfake140k_valid_csv=DATASET_PATHS['140k']['val_csv'],
            realfake140k_test_csv=DATASET_PATHS['140k']['test_csv'],
            realfake140k_root_dir=DATASET_PATHS['140k']['root_dir'],
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"  Skipping 140k dataset: {e}")

    try:
        datasets['190k'] = Dataset_selector(
            dataset_mode='190k',
            realfake190k_root_dir=DATASET_PATHS['190k']['root_dir'],
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"  Skipping 190k dataset: {e}")

    if not datasets:
        raise RuntimeError("No datasets loaded — check DATASET_PATHS in config.py")

    # ── Run pipeline ──
    student_models, results = run_pipeline(
        teacher_paths=TEACHER_PATHS,
        datasets=datasets,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        alpha=LOGITS_ALPHA,
        beta=LOGITS_BETA,
        rkd_dist_w=RKD_DISTANCE_WEIGHT,
        rkd_angle_w=RKD_ANGLE_WEIGHT,
        device=DEVICE,
        save_dir=SAVE_DIR,
        multi_teacher=MULTI_TEACHER,
        train_dataset=TRAIN_DATASET,
    )

    print("\nDone! Models saved to:", SAVE_DIR)


if __name__ == '__main__':
    main()
