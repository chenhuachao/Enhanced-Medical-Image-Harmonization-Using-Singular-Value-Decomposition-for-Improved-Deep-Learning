""" The deep learning models commonly used,
    which is developed via tensorflow.

    Author: Kuang Xihe
    Date: 2022/12/26
    Version: 2.0
"""

from tensorflow.keras.layers import Input, Conv2D, Conv3D
from tensorflow.keras.models import Model

from .blocks import *


def unet(ndim: int, b_num: int, k_num_list: list, use_bn: bool, in_ch_num: int, out_ch_num: [int, None]):
    """ 2D/3D U-Net for segmentation
        :param ndim: the number of dimension of input, 2 or 3
        :param b_num: the number of convolutional blocks of each level
        :param k_num_list: the number of kernel numbers in each level
        :param use_bn: if the model use batch normalization
        :param in_ch_num: the number of channel of input
        :param out_ch_num: the number of channel of output, if is None, only output the feature map
    """

    assert ndim in [2, 3]

    tensor_record = {}
    if ndim == 2:
        x = Input(shape=(None, None, in_ch_num))
    else:
        x = Input(shape=(None, None, None, in_ch_num))
    tensor_record['input'] = x

    # down sample
    for idx in range(len(k_num_list) - 1):
        k_num = k_num_list[idx]
        conv_block = ConvRele(ndim=ndim, k_num=k_num, use_bn=use_bn)
        down_block = DownSampleConv(ndim=ndim, conv_block=conv_block, b_num=b_num, down_size=2,
                                    block_name='Down_{}'.format(idx))
        x, fea = down_block(x)
        tensor_record['Down_{}'.format(idx)] = fea
    #
    conv_block = ConvRele(ndim=ndim, k_num=k_num_list[-1], use_bn=use_bn, block_name='Conv')
    for n in range(b_num):
        conv_block.rename('Conv_{}'.format(n))
        x = conv_block(x)
    # up sample
    for idx in range(len(k_num_list) - 1):
        k_num = k_num_list[len(k_num_list) - idx - 2]
        conv_block = ConvRele(ndim=ndim, k_num=k_num, use_bn=use_bn)
        up_block = UpSample(ndim=ndim, conv_block=conv_block, b_num=b_num, up_size=2,
                            block_name='Up_{}'.format(len(k_num_list) - idx - 2))
        x = up_block([x, tensor_record['Down_{}'.format(len(k_num_list) - idx - 2)]])
    #
    if out_ch_num is not None:
        x = ConvSoftmax(ndim=ndim, k_num=out_ch_num, k_size=1, use_bn=False, block_name='Out')(x)

    model = Model(inputs=tensor_record['input'], outputs=x)
    return model


def main():
    model = unet(ndim=2, b_num=2, k_num_list=[32, 64, 128, 256], use_bn=True, in_ch_num=1, out_ch_num=2)
    model.summary()
    print(model.input_shape)
    print(model.output_shape)
    model.save('test_unet_model.h5')


if __name__ == '__main__':
    main()
