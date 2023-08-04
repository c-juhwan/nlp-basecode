# Standard Library Modules
import os
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from PIL import Image
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path:str) -> None:
        super(CustomDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.num_classes = data_['num_classes']

        """
        https://pytorch.org/vision/stable/models.html
        Every pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        You can use the following transform to normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        """
        self.transform = transforms.Compose([
            transforms.Resize((args.image_resize_size, args.image_resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # Normalize with predefined mean & std following torchvision documents
                                (0.229, 0.224, 0.225))
        ])

        for idx in tqdm(range(len(data_['images'])), desc=f'Loading data from {data_path}'):
            PIL_images = data_['images'][idx].convert('RGB')
            torch_image = self.transform(PIL_images)

            self.data_list.append({
                'images': torch_image,
                'labels': data_['labels'][idx],
                'index': idx
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    image = torch.stack([d['images'] for d in data])
    label = torch.tensor([d['labels'] for d in data], dtype=torch.long)
    indices = torch.tensor([d['index'] for d in data], dtype=torch.long)

    datas_dict = {
        'images': image,
        'labels': label,
        'index': indices
    }

    return datas_dict
