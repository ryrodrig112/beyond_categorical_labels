from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

class ImagesWHighDimLabels(Dataset):
    def __init__(self, img_path: str, cat_label_path: str, high_dim_labels_path: str, label_map: dict, transform=None):
        self.images = np.load(img_path)
        self.cat_labels = np.load(cat_label_path)
        self.high_dim_labels = np.load(high_dim_labels_path)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.cat_labels)

    def display_item(self, idx):
        img = self.images[idx]
        cat_label = self.cat_labels[idx]
        label_text = self.label_map[cat_label]

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(label_text, fontsize=10)
        plt.show()

    def display_item_w_label(self, idx):
        img = self.images[idx]
        cat_label = self.cat_labels[idx]
        label_text = self.label_map[cat_label]

        fig, ax = plt.subplots(1, 2, figsize=(4, 2))

        ax[0].axis('off')
        ax[0].imshow(img)
        ax[0].set_title(label_text, fontsize=10)

        ax[1].axis('off')
        ax[1].imshow(self.high_dim_labels[cat_label])
        ax[1].set_title("Spectrogram", fontsize=10)
        plt.show()

    def __getitem__(self, idx):
        image = self.images[idx]
        cat_label = self.cat_labels[idx]
        high_dim_label = self.high_dim_labels[cat_label]
        if self.transform:
            image = self.transform(image)

        return image, high_dim_label, cat_label


class CIFAR10Extended(CIFAR10): #extend CIFAR 10 for high dim data
    def __init__(self, high_dim_labels_path, label_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_dim_labels = np.load(high_dim_labels_path)
        self.label_map = label_map

    def __getitem__(self, index):
        img, cat_label = super().__getitem__(index)
        high_dim_label = self.high_dim_labels[cat_label]
        return img, high_dim_label, cat_label

class CIFAR10DLGetter: # use this to create data loaders
    def __init__(self, train_pct, val_pct, batch_size, label_type: str, high_dim_label_path=None, label_map=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
        self.batch_size = batch_size
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=mean,
                                 std=std)])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])
        test_transform = val_transform

        if label_type == "categorical":
            self.train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            self.val_dataset = CIFAR10(root='./data', train=True, download=True, transform=val_transform)
            self.test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif label_type == "high_dim":
            self.train_dataset = CIFAR10Extended(high_dim_label_path, label_map, root='./data', train=True, download=True, transform=train_transform)
            self.val_dataset = CIFAR10Extended(high_dim_label_path, label_map, root='./data', train=True, download=True, transform=val_transform)
            self.test_dataset = CIFAR10Extended(high_dim_label_path, label_map, root='./data', train=False, download=True, transform=test_transform)

        # Calculate sizes of each split
        total_size = len(self.train_dataset)
        train_size = int(total_size * train_pct)
        val_size = int(total_size * val_pct)
        ignore_size = total_size - train_size - val_size

        # Random, non-contiguous split
        indices = torch.randperm(total_size).tolist()
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:train_size + val_size]
        self.high_dim_labels = None
        if label_type == "high_dim":
            self.high_dim_labels = np.load(high_dim_label_path)


    def get_trainloader(self):
        train_subset = Subset(self.train_dataset, self.train_indices)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def get_valloader(self):
        val_subset = Subset(self.val_dataset, self.val_indices)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        return val_loader

    def get_testloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader

def main():
    image_save_dir = "../data/imgs/cifar10"
    label_save_dir = "../data/labels/categorical/"
    image_file_names = ["cifar10_train_img.npy", "cifar10_test_img.npy"]
    label_file_names = ["cifar10_training.npy", "cifar10_test.npy"]

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]

    classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    label_map = {idx: label for idx, label in enumerate(classes)}

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    img_path = os.path.join(image_save_dir, image_file_names[0])
    cat_label_path = os.path.join(label_save_dir, label_file_names[0])
    high_dim_labels_path = "../data/labels/high_dim/cifar10_speech.npy"

    traindata = ImagesWHighDimLabels(img_path, cat_label_path, high_dim_labels_path, label_map,
                                          transform=transform)

    traindata.display_item(0)
    traindata.display_item_w_label(1)


if __name__ == "__main__":
    main()
