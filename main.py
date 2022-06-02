import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def download_datasets():
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
    return train_data, test_data


def show_datasets_image(datasets):
    labels = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
    }
    fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        index = torch.randint(len(datasets), size=(1,)).item()
        image, label = datasets[index]
        fig.add_subplot(rows, cols, i)
        plt.title(labels[label])
        plt.axis('off')
        plt.imshow(image.squeeze(), cmap='gray')
    plt.show()


def main():
    train_data, test_data = download_datasets()
    show_datasets_image(train_data)


if __name__ == '__main__':
    main()
