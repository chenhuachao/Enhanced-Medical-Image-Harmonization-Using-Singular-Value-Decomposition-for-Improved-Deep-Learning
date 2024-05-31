import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from deep_learning_tf.train import *
from deep_learning_tf.models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def nor_img(img):
    return (img - img.min()) / (img.max() - img.min())


def main():

    seg_dir = "/home/sun/stream_infer/task/eye/image/{}/{}.1.jpg"
    img_dir = "/home/sun/stream_infer/task/eye/image/{}/{}.jpg"
    
    model = unet(
        ndim=2,
        b_num=2,
        k_num_list=[64, 128, 256, 512],  # channel number for each level
        use_bn=True,
        in_ch_num=3,  # input channels
        out_ch_num=2,
    )  # output channels
    
    #model = tf.keras.models.load_model("model_E_120")
    op = tf.keras.optimizers.Adam(learning_rate=0.003)  # optimizer
    for e in range(1, 31):  # Training Epoch
        print(e)
        patch_list = []
        for i in range(1, 151):  # Training Set ID 75%
            img = cv2.imread(img_dir.format(i, i))
            img = cv2.resize(img, (1024, 1024))  # 调整图像尺寸为1024x1024
            img = nor_img(img)

            seg = cv2.imread(seg_dir.format(i, i))
            seg = cv2.resize(seg, (1024, 1024))  # 调整seg图像尺寸为1024x1024
            seg = seg.min(axis=-1)
            seg = np.array(1 * (seg > 230), dtype=np.uint8)

            data = np.concatenate(
                (img, 1 - np.expand_dims(seg, axis=-1), np.expand_dims(seg, axis=-1)),
                axis=-1,
            )
            patch = get_patch(
                data, patch_size=(128, 128), patch_num=64
            )  # patch daxiao patch_num!
            patch_list += [patch]
            print(i, data.shape, patch.shape)

        data = np.concatenate(tuple(patch_list), axis=0)
        data = data.astype(np.float32)
        print(data.shape)

        for _ in range(5):
            np.random.shuffle(data)
            model, op = train(
                model, op, img=data[:, :, :, :3], seg=data[:, :, :, 3:5], batch_size=4
            )

        model.save("/home/sun/stream_infer/task/eye/model/base_model_{}".format(e + 1))

        # img = cv2.imread(img_dir.format(21, 21))
        # img = nor_img(img)
        # out = model(np.expand_dims(img[1000:1256, -266:-10, :], 0)).numpy()
        # print(out.shape)
        # cv2.imwrite('{}.png'.format(e), np.hstack((img[1000:1256, -266:-10, :],
        #                                            out[0, :, :, 0:1].repeat(3, axis=-1))) * 255)


if __name__ == "__main__":
    main()
