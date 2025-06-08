from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DeblurDataset(Dataset):
    """Dataset for image deblurring task"""
    
    def __init__(self, data_path, image_size=256, is_training=True):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.is_training = is_training
        
        # Get paths
        self.blurred_path = self.data_path / 'blurred'
        self.sharp_path = self.data_path / 'sharp'
        
        if not self.blurred_path.exists() or not self.sharp_path.exists():
            raise FileNotFoundError(f"Missing blurred/ or sharp/ directories in {data_path}")
        
        # Get matching files
        self.image_files = self._get_matching_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No matching image pairs found in {data_path}")
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"Loaded {len(self.image_files)} image pairs from {data_path}")
    
    def _get_matching_files(self):
        """Get files that exist in both directories"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        blurred_files = {f.name for f in self.blurred_path.iterdir() 
                        if f.suffix.lower() in extensions}
        sharp_files = {f.name for f in self.sharp_path.iterdir() 
                      if f.suffix.lower() in extensions}
        return sorted(list(blurred_files & sharp_files))
    
    def _get_transforms(self):
        """Get data transforms"""
        if self.is_training:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'sharp': 'image'})
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'sharp': 'image'})
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load images
        blurred_img = cv2.imread(str(self.blurred_path / filename))
        sharp_img = cv2.imread(str(self.sharp_path / filename))
        
        # Convert BGR to RGB
        blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=blurred_img, sharp=sharp_img)
        
        return {
            'blurred': transformed['image'],
            'sharp': transformed['sharp'],
            'filename': filename
        }


def get_dataloaders(train_path, val_path, batch_size=8, image_size=256, num_workers=4):
    """Create dataloaders"""
    train_dataset = DeblurDataset(train_path, image_size, is_training=True)
    val_dataset = DeblurDataset(val_path, image_size, is_training=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader