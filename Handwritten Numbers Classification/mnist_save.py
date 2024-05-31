#!/usr/bin/env python

import torch
import torchvision
import os
from torchvision import transforms
import torch.utils.data as data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools

# 查看部分mnist图像
def show():
    plt.figure(figsize=(16, 9))
    for i, item in enumerate(itertools.islice(train_loader, 2, 12)):
        plt.subplot(2, 5, i+1)
        img, label = item
        img = img[0].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Move channel dimension to last
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    train_data = torchvision.datasets.ImageFolder('/home/gem/Harry/C_To_SVD_USPS_use_USPS_0817_net/train', transform=transforms.ToTensor())
    test_data = torchvision.datasets.ImageFolder('/home/gem/Harry/C_To_SVD_USPS_use_USPS_0817_net/test', transform=transforms.ToTensor())
    train_loader = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    train_total = len(train_loader)
    test_total = len(test_loader)
    labels = train_data.targets
    print(labels)
    print(train_total, test_total)
    dataiter = iter(train_loader)
    images, labs = dataiter.__next__()
    print(type(images), type(labs))
    print(images.shape, labs)
    show()
