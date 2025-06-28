import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SEN12MS(Dataset):
    def __init__(self, root, split='train', bands=None, 
                 patch_size=16, num_samples=None, transform=None):
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.bands = bands or {
            'SAR': ['VV', 'VH'],
            'OPTICAL': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
        }
        
        self.samples = self._load_samples()
        if num_samples:
            self.samples = self.samples[:num_samples]
            
    def _load_samples(self):
        samples = []
        base_path = os.path.join(self.root, self.split)
        
        for scene_id in os.listdir(base_path):
            scene_path = os.path.join(base_path, scene_id)
            if not os.path.isdir(scene_path): continue
                
            patch_ids = [f.split('.')[0] for f in os.listdir(
                os.path.join(scene_path, 's1')) if f.endswith('.tif')]
                
            for pid in patch_ids:
                samples.append({
                    'sar': os.path.join(scene_path, 's1', f'{pid}.tif'),
                    'optical': os.path.join(scene_path, 's2', f'{pid}.tif'),
                    'label': os.path.join(scene_path, 'lc', f'{pid}.tif')
                })
                
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sar = np.array(Image.open(sample['sar']))
        sar = sar[:, :, :len(self.bands['SAR'])]
        
        optical = np.array(Image.open(sample['optical']))
        optical = optical[:, :, :len(self.bands['OPTICAL'])]
        
        sar = self._preprocess_sar(sar)
        optical = self._preprocess_optical(optical)
        
        if self.transform:
            sar, optical = self.transform(sar, optical)
            
        return sar, optical
    
    def _preprocess_sar(self, sar):
        sar = np.log1p(sar)
        sar = (sar - sar.min()) / (sar.max() - sar.min())
        return sar.transpose(2, 0, 1)  
    
    def _preprocess_optical(self, optical):
        optical = np.clip(optical / 10000.0, 0, 1)
        return optical.transpose(2, 0, 1) 
