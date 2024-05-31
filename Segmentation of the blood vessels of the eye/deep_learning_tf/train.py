""" The function for model training,
    which is developed via tensorflow.

    Author: Kuang Xihe
    Date: 2022/12/26
    Version: 2.0
"""

import numpy as np
import tensorflow as tf


def pixel_classification_loss(out, seg):
    """ pixel classification loss based on cross entropy. """
    loss = tf.math.log(out + 1e-6) * seg
    loss = tf.math.reduce_sum(loss, axis=[0, 1, 2]) / (tf.math.reduce_sum(seg, axis=[0, 1, 2]) + 1)
    loss = tf.math.reduce_mean(loss)

    return -loss


def train(model, optimizer, img: np.ndarray, seg: np.ndarray, batch_size: int):
    """ training the model for segmentation task in an end-to-end way

    :param model: the deep learning model
    :param img: the input images or features, channel last
    :param seg: the ground truth, 1st channel is background
    :param batch_size: batch size
    :param optimizer: optimizer for training
    :return: trained model and optimizer
    """

    def _train_batch(_img_b: np.ndarray, _seg_b: np.ndarray):
        with tf.GradientTape() as tape:
            out = model(_img_b, training=True)
            loss = pixel_classification_loss(out, _seg_b)
            record = float(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return record

    batch_num = img.shape[0] // batch_size
    rec_list = []
    for i in range(batch_num):
        rec = _train_batch(_img_b=img[i * batch_size:(i + 1) * batch_size],
                           _seg_b=seg[i * batch_size:(i + 1) * batch_size])
        rec_list += [rec]
        print('{}/{}: Loss: {}'.format(i + 1, batch_num, rec), end='\r')
    print()
    print('Ave: {}'.format(np.array(rec_list).mean(axis=0)))

    return model, optimizer


def get_patch(image, patch_size, patch_num):
    xs = np.random.randint(0, image.shape[0] - patch_size[0], patch_num)
    ys = np.random.randint(0, image.shape[1] - patch_size[1], patch_num)
    patches = []
    for i in range(patch_num):
        patches += [np.expand_dims(image[xs[i]:xs[i] + patch_size[0], ys[i]:ys[i]+patch_size[1], :], axis=0)]
    patches = np.concatenate(tuple(patches), axis=0)
    return patches

