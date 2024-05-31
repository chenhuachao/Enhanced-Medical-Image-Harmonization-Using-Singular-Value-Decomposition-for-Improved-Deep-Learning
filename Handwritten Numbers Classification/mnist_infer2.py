import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
from CNN_new import CnnNet  

'''
    手写数字预测+特征提取

'''


# 预测类
class Pred:
    def __init__(self):
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        model_path = '/home/gem/Harry/MNIST_Model/MNIST_checkpoints/Har_aug_2/model_epoch_19.pth'
        self.model = CnnNet(classes=10)  # 实例化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型参数
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)

    # 预测
    def predict(self, img_path):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transform(img)
        img = img.view(1, 3, 28, 28).to(self.device)
        output = self.model(img)
        output = torch.softmax(output, dim=1)
        # 每个预测值的概率
        probability = output.cpu().detach().numpy()[0]
        # 找出最大概率值的索引
        output = torch.argmax(output, dim=1)
        index = output.cpu().numpy()[0]
        # 预测结果
        pred = self.labels[index]
        return pred, probability[index]

    # 批量预测并计算准确率
    def predict_and_compute_accuracy(self, img_folder):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        correct = 0
        total = 0

        with torch.no_grad():
            for class_name in os.listdir(img_folder):
                class_folder = os.path.join(img_folder, class_name)
                if os.path.isdir(class_folder):
                    class_label = int(class_name)
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        img = transform(img)
                        img = img.view(1, 3, 28, 28).to(self.device)
                        output = self.model(img)
                        output = torch.softmax(output, dim=1)
                        _, predicted = torch.max(output, 1)
                        total += 1
                        if class_label == predicted.item():
                            correct += 1

        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy


if __name__ == '__main__':
    img_folder = '/home/gem/Harry/USPS/usps/test'  # 更新为包含测试图像的文件夹路径
    pred = Pred()

    # 批量预测并计算准确率
    acc = pred.predict_and_compute_accuracy(img_folder)
    print(f'Test Accuracy: {acc:.4f}')
