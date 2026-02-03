from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import default_collate
from torchvision.transforms import v2
import torch

class CustomImageDataset:
    def __init__(self, data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = ImageFolder(data_dir)
        self.classes = self.images.classes
        self.class_to_idx = self.images.class_to_idx
        self.num_classes = len(self.classes)
        self.targets = [label for _, label in self.images.samples]
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, label = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
        
def get_image_dataset(args):
    #setting args for default values
    if not hasattr(args, 'data_dir'):
        raise ValueError('data_dir is required')
    if not hasattr(args, 'batch_size'):
        args.batch_size = 1
    if not hasattr(args, 'num_workers'):
        args.num_workers = 1
    if not hasattr(args, 'transform'):
        args.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    if not hasattr(args, 'shuffle'):
        args.shuffle = True

    
    dataset = CustomImageDataset(args.data_dir, transform=args.transform)
    loader =  DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                          num_workers=args.num_workers, persistent_workers=args.persistent_workers,
                          pin_memory=True)
    return loader


#main for assertion
if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--transform', type=str, default='default')
    parser.add_argument('--persistent_workers', type=bool, default=False)

    args = parser.parse_args()
    args.transform = v2.Compose([v2.ToImage(),v2.Resize((224, 224)), v2.ToDtype(torch.float16, scale=True)])
    loader = get_image_dataset(args)
    for batch in loader:
        print(batch)
        break