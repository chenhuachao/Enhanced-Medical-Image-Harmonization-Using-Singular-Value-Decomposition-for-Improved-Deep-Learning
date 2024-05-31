import torch
import torch.nn
import onnx
from torchvision import transforms
#调用自己的模型
from CNN import CnnNet
import torch.nn as nn
 
# 定义求导函数
def get_Variable(x):
    x = torch.autograd.Variable(x)  # Pytorch 的自动求导
    # 判断是否有可用的 GPU
    return x.cuda() if torch.cuda.is_available() else x
 
 
# 判断是否GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device('cpu')

# 定义网络
model = CnnNet()
 
# 直接加载模型
loaded_model = torch.load('/home/gem/Harry/USPS_ICCV2_Compare/checkpoints/last.pt', map_location='cuda:0')
model.load_state_dict(loaded_model['state_dict'])
model.eval()
 
input_names = ['input']
output_names = ['output']
 
# x = torch.randn(1,3,32,32,requires_grad=True)
x = torch.randn(1, 3, 28, 28, requires_grad=True)  # 这个要与你的训练模型网络输入一致。
 
torch.onnx.export(model, x, '/home/gem/Harry/ONNX/Pytorch/pytorch_USPSICCV2_model.onnx', input_names=input_names, output_names=output_names, verbose='True')
 

