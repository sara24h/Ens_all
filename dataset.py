# dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        # خروجی به صورت long برای کاربردهای دسته‌بندی و تقطیر دانش
        return image, torch.tensor(label, dtype=torch.long)

class Dataset_selector:
    def __init__(
        self,
        dataset_mode,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_root_dir=None,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        realfake200k_train_csv=None,
        realfake200k_val_csv=None,
        realfake200k_test_csv=None,
        realfake200k_root_dir=None,
        realfake190k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        pin_memory=True,
    ):
        if dataset_mode not in ['140k', '190k', '200k']:
            raise ValueError("Supported dataset_mode for this setup: '140k', '190k', or '200k'")
            
        self.dataset_mode = dataset_mode
        image_size = (256, 256)

        if dataset_mode == '140k':
            mean, std = (0.5207, 0.4258, 0.3806), (0.2490, 0.2239, 0.2212)
        elif dataset_mode == '200k':
            mean, std = (0.4460, 0.3622, 0.3416), (0.2057, 0.1849, 0.1761)
        elif dataset_mode == '190k':
            mean, std = (0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081)

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
        root_dir, train_data, val_data, test_data = None, None, None, None

        if dataset_mode == '140k':
            train_data = pd.read_csv(realfake140k_train_csv)
            val_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')
            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        elif dataset_mode == '200k':
            train_data = pd.read_csv(realfake200k_train_csv)
            val_data = pd.read_csv(realfake200k_val_csv)
            test_data = pd.read_csv(realfake200k_test_csv)
            root_dir = realfake200k_root_dir

            def create_image_path(row, split):
                folder = 'real' if row['label'] in [1, 'real', 'Real'] else 'fake'
                img_name = os.path.basename(row.get('filename_clean', row.get('filename', row.get('image', row.get('path', '')))))
                return os.path.join(split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            val_data['images_id'] = val_data.apply(lambda row: create_image_path(row, 'val'), axis=1)
            test_data['images_id'] = test_data.apply(lambda row: create_image_path(row, 'test'), axis=1)

        elif dataset_mode == '190k':
            root_dir = realfake190k_root_dir
            def collect_images_from_folder(split):
                data = []
                for label in ['Real', 'Fake']:
                    folder_path = os.path.join(root_dir, split, label)
                    if not os.path.exists(folder_path):
                        raise FileNotFoundError(f"Folder not found: {folder_path}")
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(split, label, img_name)
                            data.append({'images_id': img_path, 'label': label})
                return pd.DataFrame(data)
            train_data = collect_images_from_folder('Train')
            val_data = collect_images_from_folder('Validation')
            test_data = collect_images_from_folder('Test')

        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

        self.loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_val = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_test = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
