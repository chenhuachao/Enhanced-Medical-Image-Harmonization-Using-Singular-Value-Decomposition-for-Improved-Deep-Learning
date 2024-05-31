import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from CNN_new import CnnNet  
from mnist_dataset import MnistData  
import random
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def split_data(files):
    """
    将数据集划分为训练集和验证集
    :param files: 数据文件列表
    :return: 分割后的训练集和验证集
    """
    random.shuffle(files)
    ratio = 0.9  # 定义训练集和验证集的比例
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data

def train(model, loss_func, optimizer, checkpoints, epochs, device):
    print('start train...')
    best_acc = 0
    best_epoch = 0
    best_model_path = ""

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss, train_acc = 0, 0
        all_preds, all_targets = [], []

        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item()
            train_acc += (preds == labels).sum().item()

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item()
                val_acc += (preds == labels).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # 计算并保存混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('Ground_truth')
        plt.savefig(f"{checkpoints}/confusion_matrix_epoch_{epoch}.png")
        plt.close()

        epoch_train_acc = train_acc / len(train_data.dataset)
        epoch_val_acc = val_acc / len(val_data.dataset)

        print(f'Epoch {epoch} | Train Loss : {train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | Time: {time.time() - start_time:.2f}s')

        # 保存每个epoch的模型
        epoch_model_path = f"{checkpoints}/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_model_path)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = epoch
            best_model_path = epoch_model_path

    print(f'Training is complete. The best models are located in Epoch {best_epoch}, ACC is {best_acc:.4f}, Model path: {best_model_path}')

if __name__ == '__main__':
    bs = 32
    lr = 0.01
    epochs = 50
    checkpoints = '/home/gem/Harry/MNIST_Model/MNIST_checkpoints/svd_harmoniozation'
    os.makedirs(checkpoints, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])

    labels = list(range(10))
    base_dir = '/home/gem/Harry/MNIST/mnist_svd_harmoniozation/train'
    imgs = []
    for label in labels:
        label_dir = os.path.join(base_dir, str(label))
        images = os.listdir(label_dir)
        for img in images:
            img_path = os.path.join(label_dir, img)
            imgs.append((img_path, label))

    trains, vals = split_data(imgs)
    train_dataset = MnistData(trains, transform=transform)
    train_data = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataset = MnistData(vals, transform=transform)
    val_data = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    model = CnnNet(classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train(model, loss_func, optimizer, checkpoints, epochs, device)
