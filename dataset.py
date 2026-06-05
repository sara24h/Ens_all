"""
Dataset loader — supports 200k, 140k, 190k deepfake datasets.

This file contains the FaceDataset class and Dataset_selector
for loading and preprocessing the deepfake image datasets.

Usage:
    from dataset import Dataset_selector

    ds = Dataset_selector(
        dataset_mode='200k',
        realfake200k_train_csv='...',
        realfake200k_val_csv='...',
        realfake200k_test_csv='...',
        realfake200k_root_dir='...',
        train_batch_size=32,
    )
    # Access: ds.loader_train, ds.loader_val, ds.loader_test
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class FaceDataset(Dataset):
    """Deepfake face image dataset."""

    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {
            1: 1, 0: 0,
            'real': 1, 'fake': 0,
            'Real': 1, 'Fake': 0,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.data[self.img_column].iloc[idx]
        )
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")

        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)


class Dataset_selector:
    """
    Unified dataset selector for multiple deepfake datasets.

    Supported modes: 'rvf10k', '140k', '190k', '200k', '330k'

    After construction, access:
        .loader_train   — training DataLoader
        .loader_val     — validation DataLoader
        .loader_test    — test DataLoader
    """

    def __init__(
        self,
        dataset_mode,
        # RVF10K
        rvf10k_train_csv=None, rvf10k_valid_csv=None, rvf10k_root_dir=None,
        # 140k
        realfake140k_train_csv=None, realfake140k_valid_csv=None,
        realfake140k_test_csv=None, realfake140k_root_dir=None,
        # 200k
        realfake200k_train_csv=None, realfake200k_val_csv=None,
        realfake200k_test_csv=None, realfake200k_root_dir=None,
        # 190k
        realfake190k_root_dir=None,
        # 330k
        realfake330k_root_dir=None,
        # DataLoader settings
        train_batch_size=32, eval_batch_size=32,
        num_workers=8, pin_memory=True, ddp=False,
    ):
        valid_modes = ['hardfake', 'rvf10k', '140k', '190k', '200k', '330k']
        if dataset_mode not in valid_modes:
            raise ValueError(f"dataset_mode must be one of {valid_modes}")
        self.dataset_mode = dataset_mode

        image_size = (256, 256) if dataset_mode != 'hardfake' else (300, 300)

        # Normalization stats per dataset
        stats = {
            'rvf10k':   ((0.5212, 0.4260, 0.3811), (0.2486, 0.2238, 0.2211)),
            '140k':     ((0.5207, 0.4258, 0.3806), (0.2490, 0.2239, 0.2212)),
            '200k':     ((0.4460, 0.3622, 0.3416), (0.2057, 0.1849, 0.1761)),
            '190k':     ((0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081)),
            'hardfake': ((0.4923, 0.4042, 0.3624), (0.2446, 0.2198, 0.2141)),
            '330k':     ((0.4923, 0.4042, 0.3624), (0.2446, 0.2198, 0.2141)),
        }
        mean, std = stats.get(dataset_mode, stats['hardfake'])

        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        img_column = 'path' if dataset_mode == '140k' else 'images_id'

        root_dir = None
        train_data = val_data = test_data = None

        # ── RVF10K ──
        if dataset_mode == 'rvf10k':
            if not all([rvf10k_train_csv, rvf10k_valid_csv, rvf10k_root_dir]):
                raise ValueError("rvf10k requires train_csv, valid_csv, root_dir")
            train_data = pd.read_csv(rvf10k_train_csv)
            train_data['images_id'] = train_data.apply(
                lambda r: _rvf_path(r, 'train'), axis=1
            )
            valid_data = pd.read_csv(rvf10k_valid_csv)
            valid_data['images_id'] = valid_data.apply(
                lambda r: _rvf_path(r, 'valid'), axis=1
            )
            val_data, test_data = train_test_split(
                valid_data, test_size=0.5,
                stratify=valid_data['label'], random_state=3407,
            )
            val_data = val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            root_dir = rvf10k_root_dir

        # ── 140k ──
        elif dataset_mode == '140k':
            if not all([realfake140k_train_csv, realfake140k_valid_csv,
                        realfake140k_test_csv, realfake140k_root_dir]):
                raise ValueError("140k requires train_csv, valid_csv, test_csv, root_dir")
            train_data = pd.read_csv(realfake140k_train_csv).sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)
            val_data = pd.read_csv(realfake140k_valid_csv).sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)
            test_data = pd.read_csv(realfake140k_test_csv).sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)
            root_dir = os.path.join(
                realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake'
            )

        # ── 200k ──
        elif dataset_mode == '200k':
            if not all([realfake200k_train_csv, realfake200k_val_csv,
                        realfake200k_test_csv, realfake200k_root_dir]):
                raise ValueError("200k requires train_csv, val_csv, test_csv, root_dir")
            train_data = pd.read_csv(realfake200k_train_csv)
            val_data = pd.read_csv(realfake200k_val_csv)
            test_data = pd.read_csv(realfake200k_test_csv)
            root_dir = realfake200k_root_dir

            for df, split in [(train_data, 'train'),
                              (val_data, 'val'),
                              (test_data, 'test')]:
                df['images_id'] = df.apply(
                    lambda r: _200k_path(r, split), axis=1
                )

        # ── 190k ──
        elif dataset_mode == '190k':
            if not realfake190k_root_dir:
                raise ValueError("190k requires root_dir")
            root_dir = realfake190k_root_dir
            train_data = _collect_folder(root_dir, 'Train')
            val_data   = _collect_folder(root_dir, 'Validation')
            test_data  = _collect_folder(root_dir, 'Test')

        # ── 330k ──
        elif dataset_mode == '330k':
            if not realfake330k_root_dir:
                raise ValueError("330k requires root_dir")
            root_dir = realfake330k_root_dir
            train_data = _collect_folder(root_dir, 'train').sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)
            val_data = _collect_folder(root_dir, 'valid').sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)
            test_data = _collect_folder(root_dir, 'test').sample(
                frac=1, random_state=3407
            ).reset_index(drop=True)

        # ── Debug info ──
        print(f"\n[{dataset_mode}] dataset statistics:")
        print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
        for split_name, data in [('train', train_data),
                                  ('val', val_data),
                                  ('test', test_data)]:
            print(f"  {split_name} labels:\n{data['label'].value_counts().to_string()}")

        # ── Create DataLoaders ──
        ds_train = FaceDataset(train_data, root_dir, transform_train, img_column)
        ds_val   = FaceDataset(val_data,   root_dir, transform_test,  img_column)
        ds_test  = FaceDataset(test_data,  root_dir, transform_test,  img_column)

        loader_kwargs = dict(
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if ddp:
            from torch.utils.data.distributed import DistributedSampler
            self.loader_train = DataLoader(
                ds_train, sampler=DistributedSampler(ds_train, shuffle=True),
                **loader_kwargs,
            )
            self.loader_val = DataLoader(
                ds_val, sampler=DistributedSampler(ds_val, shuffle=False),
                batch_size=eval_batch_size, num_workers=num_workers,
                pin_memory=pin_memory,
            )
            self.loader_test = DataLoader(
                ds_test, sampler=DistributedSampler(ds_test, shuffle=False),
                batch_size=eval_batch_size, num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            self.loader_train = DataLoader(ds_train, shuffle=True, **loader_kwargs)
            self.loader_val = DataLoader(
                ds_val, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            self.loader_test = DataLoader(
                ds_test, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        print(f"  DataLoaders ready — Train: {len(self.loader_train)} batches, "
              f"Val: {len(self.loader_val)}, Test: {len(self.loader_test)}")


# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

def _rvf_path(row, split):
    folder = 'fake' if row['label'] == 0 else 'real'
    name = os.path.basename(row['id'])
    if not name.endswith('.jpg'):
        name += '.jpg'
    return os.path.join('rvf10k', split, folder, name)


def _200k_path(row, split):
    folder = 'real' if row['label'] in [1, 'real', 'Real'] else 'fake'
    for key in ('filename_clean', 'filename', 'image', 'path'):
        if key in row:
            name = os.path.basename(row[key])
            break
    else:
        name = ''
    return os.path.join(split, folder, name)


def _collect_folder(root_dir, split):
    data = []
    for label in ['Real', 'Fake']:
        folder_path = os.path.join(root_dir, split, label)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({
                    'images_id': os.path.join(split, label, img_name),
                    'label': label,
                })
    return pd.DataFrame(data)
