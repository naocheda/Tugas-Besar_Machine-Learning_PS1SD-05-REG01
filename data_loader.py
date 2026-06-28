# Referensi PyTorch Dataset & DataLoader: https://pytorch.org/docs/stable/data.html
# Referensi Torchvision Transforms: https://pytorch.org/vision/stable/transforms.html

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AlzheimerDataset(Dataset):
    def __init__(self, csv_file, root_dir, split_type, transform=None):
        df = pd.read_csv(csv_file)
        # Memfilter baris berdasarkan 'train' atau 'test'
        self.data = df[df['Split'] == split_type].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'Non_Demented': 0, 'Very_Mild_Demented': 1, 'Mild_Demented': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Split'], row['Disease'], row['Filename'])
        # MobileNetV2 dari Kaggle membutuhkan format 3-channel RGB
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['Disease']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(csv_path, root_dir, batch_size=32):
    # Menggunakan nilai normalisasi standar ImageNet sesuai kode Kaggle
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = AlzheimerDataset(csv_path, root_dir, split_type='train', transform=train_transform)
    test_ds = AlzheimerDataset(csv_path, root_dir, split_type='test', transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
