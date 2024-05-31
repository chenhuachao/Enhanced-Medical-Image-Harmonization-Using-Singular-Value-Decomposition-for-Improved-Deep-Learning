import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_curve, auc, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# 定义测试图片目录
TEST_DIR_CANCER = '/home/gem/Harry/Breast/RSNA_data_test/cancer/'
TEST_DIR_NOCANCER = '/home/gem/Harry/Breast/RSNA_data_test/nocancer/'

# 加载模型
model_path = '/home/gem/Harry/Breast_Code/model/BreastCancer_best_model_baseline10.h5'
final_model = load_model(model_path)

# 函数：加载图像，进行预处理，并预测
def load_and_preprocess_image(path, size):
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.NEAREST)
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    img = preprocess_input(img)
    return img

# 函数：预测并存储结果
def predict_and_store_results(directory, label, size):
    images = os.listdir(directory)
    y_true = [label] * len(images)
    y_pred = []
    for image in images:
        img = load_and_preprocess_image(os.path.join(directory, image), size)
        prediction = final_model.predict(img)
        y_pred.append(prediction[0][0])  # Assuming [1] is the probability of 'cancer'
    return y_true, y_pred

# 预测
IMG_SIZE = (512, 512)  # 图像大小
y_true_cancer, y_pred_cancer = predict_and_store_results(TEST_DIR_CANCER, 1, IMG_SIZE)
y_true_nocancer, y_pred_nocancer = predict_and_store_results(TEST_DIR_NOCANCER, 0, IMG_SIZE)

# 合并结果
y_true = y_true_cancer + y_true_nocancer
y_pred = y_pred_cancer + y_pred_nocancer

# 计算指标
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
accuracy = accuracy_score(y_true, [1 if x > 0.5 else 0 for x in y_pred])

print(f'Accuracy: {accuracy:.4f}')
print(f'AUC: {roc_auc:.4f}')

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
