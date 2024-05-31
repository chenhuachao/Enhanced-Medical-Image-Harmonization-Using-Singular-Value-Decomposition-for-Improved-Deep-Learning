import numpy as np
import torch
import os
from openvino.inference_engine import IECore
from torchvision import transforms
from PIL import Image

class PredOpenVINO:
    def __init__(self, model_xml, model_bin):
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ### 加载OpenVINO模型
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.ie = IECore()
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU", num_requests=1)
    
    ### 预处理图片
    def preprocess_image(self, img_path):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = transform(img)
        img = img.view(1, 3, 28, 28)
        return img

    def predict(self, img_path):
        img = self.preprocess_image(img_path)

        # 使用 OpenVINO 执行推理
        input_blob = next(iter(self.net.input_info))
        output_blob = next(iter(self.net.outputs))
        result = self.exec_net.infer(inputs={input_blob: img})
        output = result[output_blob]

        # 处理推理结果
        output = np.squeeze(output)
        probability = np.max(output)
        index = np.argmax(output)
        pred = self.labels[index]
        return pred, probability

    def predict_and_compute_accuracy(self, img_folder):
        correct = 0
        total = 0

        with torch.no_grad():
            for class_name in os.listdir(img_folder):
                class_folder = os.path.join(img_folder, class_name)
                if os.path.isdir(class_folder):
                    class_label = int(class_name)
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        pred, _ = self.predict(img_path)
                        total += 1
                        if class_label == pred:
                            correct += 1

        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy

if __name__ == '__main__':
    model_xml = '/home/gem/Harry/ONNX/Pytorch/model/pytorch_USPSICCV2_model.xml'
    model_bin = '/home/gem/Harry/ONNX/Pytorch/model/pytorch_USPSICCV2_model.bin'
    img_folder = '/home/gem/Harry/USPS/usps_ICCV/test'
    pred_openvino = PredOpenVINO(model_xml, model_bin)
    acc = pred_openvino.predict_and_compute_accuracy(img_folder)
    print(f'Test Accuracy: {acc:.4f}')
