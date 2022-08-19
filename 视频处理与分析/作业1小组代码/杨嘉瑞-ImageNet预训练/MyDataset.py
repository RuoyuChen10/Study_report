import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from preprocessing import parse_json


class MyDataset(Dataset):
    def __init__(self, root_path: str, train=True, image_size=256, flag=None, fold=None):
        self.root_path = root_path
        self.train_flag = train
        self.flag = flag
        self.fold = fold
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.527, 0.713, 0.680], std=[0.224, 0.208, 0.340]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path_list = os.listdir(root_path)
        if self.flag is not None:
            self.length = len(self.path_list) // 10
            if flag == 0: # train
                self.path_list = self.path_list[: self.fold * self.length] + self.path_list[(self.fold + 1) * self.length:]
            else:
                self.path_list = self.path_list[self.fold * self.length: (self.fold + 1) * self.length]

    def __getitem__(self, idx: int):
        data_path = self.path_list[idx]
        if self.train_flag is True:
            label = torch.tensor(parse_json(os.path.join(self.root_path, data_path, data_path + '.json')))
        else:
            label = None
        label = torch.as_tensor(label, dtype=torch.float32)
        image_path = os.path.join(self.root_path, data_path, 'BireView.png')
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label, image_path

    def __len__(self) -> int:
        return len(self.path_list)