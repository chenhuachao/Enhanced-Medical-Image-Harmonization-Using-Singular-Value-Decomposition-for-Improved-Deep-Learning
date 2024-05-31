import har_fix
import os
import numpy as np
from PIL import Image, ImageStat
from concurrent.futures import ThreadPoolExecutor

# 读取图片
in_path = "/home/gem/Harry/Aa——2024/final-svd-har/"
ref_img = "/home/gem/Harry/Aa——2024/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.png"
out_path = "/home/gem/Harry/Aa——2024/final-svd-har-out"

# 创建输出文件夹
if not os.path.exists(out_path):
    os.mkdir(out_path)

re_img = har_fix.imgHandle(ref_img)

# 读取当前目录下的所有文件
files = os.listdir(in_path)
for file in files:
    if file == ".DS_Store":
        continue
    print(in_path + file)
    in_img = har_fix.imgHandle(in_path + file)
    out_img = har_fix.imgMerge(in_img, re_img)
    out_img = Image.fromarray(np.rint(out_img).astype("uint8"))
    out_img.save(out_path + "/" + file)
    # out_img.save("out_" + file, out_path) 