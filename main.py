#!/usr/bin/env python
"""
Main Entry Point
================
Run all knowledge distillation training and ensemble evaluation.

Usage:
    # Train all three students
    python main.py --mode train_all \
        --teacher_140k /path/to/teacher_140k.pth \
        --teacher_190k /path/to/teacher_190k.pth \
        --teacher_200k /path/to/teacher_200k.pth

    # Run ensemble evaluation
    python main.py --mode ensemble \
        --logits_checkpoint ./checkpoints/logits_140k/best_model.pth \
        --at_checkpoint ./checkpoints/at_190k/best_model.pth \
        --rkd_checkpoint ./checkpoints/rkd_200k/best_model.pth

    # Run everything
    python main.py --mode all ...
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation for Deepfake Detection')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train_logits', 'train_at', 'train_rkd', 
                               'train_all', 'ensemble', 'all'],
                       help='What to run')
    
    # Teacher checkpoints
    parser.add_argument('--teacher_140k', type=str,
                       help='Path to teacher checkpoint for 140k dataset')
    parser.add_argument('--teacher_190k', type=str,
                       help='Path to teacher checkpoint for 190k dataset')
    parser.add_argument('--teacher_200k', type=str,
                       help='Path to teacher checkpoint for 200k dataset')
    
    # Student checkpoints (for ensemble)
    parser.add_argument('--logits_checkpoint', type=str,
                       default='/kaggle/input/models/sara24h/teacher_model_best/pytorch/default/1/teacher_model_best.pth',
                       help='Path to logits student checkpoint')
    parser.add_argument('--at_checkpoint', type=str,
                       default='/kaggle/input/datasets/sara24h/kdfs-190k-transfer-learning-data/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth',
                       help='Path to AT student checkpoint')
    parser.add_argument('--rkd_checkpoint', type=str,
                       default='/kaggle/input/datasets/sarah20079/teacher-model-best-200k/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth',
                       help='Path to RKD student checkpoint')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def run_command(cmd):
    """Run a shell command."""
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def train_logits(args):
    """Train student with Logits distillation on 140k."""
    if not args.teacher_140k:
        print("Error: --teacher_140k is required for logits training")
        return 1
    
    cmd = [
        sys.executable, 'scripts/train_logits_140k.py',
        '--teacher_checkpoint', args.teacher_140k,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--device', args.device
    ]
    return run_command(cmd)


def train_at(args):
    """Train student with AT distillation on 190k."""
    if not args.teacher_190k:
        print("Error: --teacher_190k is required for AT training")
        return 1
    
    cmd = [
        sys.executable, 'scripts/train_at_190k.py',
        '--teacher_checkpoint', args.teacher_190k,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--device', args.device
    ]
    return run_command(cmd)


def train_rkd(args):
    """Train student with RKD distillation on 200k."""
    if not args.teacher_200k:
        print("Error: --teacher_200k is required for RKD training")
        return 1
    
    cmd = [
        sys.executable, 'scripts/train_rkd_200k.py',
        '--teacher_checkpoint', args.teacher_200k,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--device', args.device
    ]
    return run_command(cmd)


def run_ensemble(args):
    """Run ensemble evaluation."""
    cmd = [
        sys.executable, 'scripts/run_ensemble.py',
        '--logits_checkpoint', args.logits_checkpoint,
        '--at_checkpoint', args.at_checkpoint,
        '--rkd_checkpoint', args.rkd_checkpoint,
        '--device', args.device
    ]
    return run_command(cmd)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Knowledge Distillation for Deepfake Detection")
    print(f"Mode: {args.mode}")
    print("=" * 70)
    
    results = {}
    
    if args.mode == 'train_logits':
        results['logits'] = train_logits(args)
    
    elif args.mode == 'train_at':
        results['at'] = train_at(args)
    
    elif args.mode == 'train_rkd':
        results['rkd'] = train_rkd(args)
    
    elif args.mode == 'train_all':
        results['logits'] = train_logits(args)
        results['at'] = train_at(args)
        results['rkd'] = train_rkd(args)
    
    elif args.mode == 'ensemble':
        results['ensemble'] = run_ensemble(args)
    
    elif args.mode == 'all':
        results['logits'] = train_logits(args)
        results['at'] = train_at(args)
        results['rkd'] = train_rkd(args)
        results['ensemble'] = run_ensemble(args)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for task, code in results.items():
        status = "SUCCESS" if code == 0 else "FAILED"
        print(f"  {task}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
