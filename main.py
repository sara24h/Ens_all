"""
main.py
=======
Entry point — builds datasets, runs the full KD + ensemble pipeline.

Usage
-----
    python main.py
"""

from config import *
from pipeline import run_full_pipeline

# Import your Dataset_selector (adjust import path as needed)
# from dataset import Dataset_selector


def build_datasets():
    """Create Dataset_selector instances from config paths."""
    datasets = {}

    # ── 200 k ───────────────────────────────────────────────────────────
    p = DATASET_PATHS["200k"]
    datasets["200k"] = Dataset_selector(
        dataset_mode="200k",
        realfake200k_train_csv=p["train_csv"],
        realfake200k_val_csv=p["val_csv"],
        realfake200k_test_csv=p["test_csv"],
        realfake200k_root_dir=p["root_dir"],
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    # ── 140 k ───────────────────────────────────────────────────────────
    p = DATASET_PATHS["140k"]
    datasets["140k"] = Dataset_selector(
        dataset_mode="140k",
        realfake140k_train_csv=p["train_csv"],
        realfake140k_valid_csv=p["valid_csv"],
        realfake140k_test_csv=p["test_csv"],
        realfake140k_root_dir=p["root_dir"],
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    # ── 190 k ───────────────────────────────────────────────────────────
    p = DATASET_PATHS["190k"]
    datasets["190k"] = Dataset_selector(
        dataset_mode="190k",
        realfake190k_root_dir=p["root_dir"],
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    return datasets


if __name__ == "__main__":
    datasets = build_datasets()

    student_models, results = run_full_pipeline(
        teacher_paths=TEACHER_PATHS,
        datasets=datasets,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        alpha=ALPHA,
        beta=BETA,
        rkd_dist_w=RKD_DISTANCE_WEIGHT,
        rkd_angle_w=RKD_ANGLE_WEIGHT,
        device=DEVICE,
        save_dir=SAVE_DIR,
        multi_teacher=MULTI_TEACHER,
        train_dataset=TRAIN_DATASET,
    )
