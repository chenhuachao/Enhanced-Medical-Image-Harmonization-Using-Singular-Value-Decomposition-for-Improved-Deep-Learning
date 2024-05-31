""" The blocks commonly used in deep learning model,
    which developed via tensorflow.

    Author: Kuang Xihe
    Date: 2022/12/24
    Version: 2.0
"""

from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.keras.layers import Conv2D, Conv3D, ReLU, Softmax, MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate


class Block:
    """ parent class of block """
    def __init__(self, block_name):
        self.block_name = block_name

    def rename(self, block_name):
        self.block_name = block_name


class ConvRele(Block):
    """ ConvRele block for feature extraction:
        concatenate (if necessary, merge multiple input tensors)
        + convolutional layer
        + batch-normalization (if necessary)
        + relu

        the input of block can be one tensor or a list of tensors with the same shape
        the output of block is a tensor with the same shape of input
        the tensor is channel last
    """

    def __init__(self, ndim: int, k_num: int, k_size: int = 3, use_bn: bool = False, block_name: str = 'ConvRelu2D'):
        """
            :param ndim: the number of dimension of input, should be 2 or 3
            :param k_num: the number kernel for convolutional layer
            :param k_size: the size of kernel for convolutional layer
            :param use_bn: flag of batch normalization for convolutional layer
            :param block_name: block name
        """

        assert ndim in [2, 3]

        super().__init__(block_name)
        self.ndim = ndim
        self.k_num = k_num
        self.k_size = k_size
        self.use_bn = use_bn

    def __call__(self, x):
        # if input multiple tensors, concatenate them, channel last
        if isinstance(x, list):
            x = Concatenate(axis=-1, name=self.block_name + '_cate')(x)
        # convolutional
        if self.ndim == 2:
            x = Conv2D(filters=self.k_num, kernel_size=self.k_size, strides=1, padding='same',
                       use_bias=(not self.use_bn),
                       name=self.block_name + '_conv')(x)
        else:
            x = Conv3D(filters=self.k_num, kernel_size=self.k_size, strides=1, padding='same',
                       use_bias=(not self.use_bn),
                       name=self.block_name + '_conv')(x)
        # batch normalization
        if self.use_bn:
            x = BatchNormalization(axis=-1, name=self.block_name + '_bn')(x)
        # activation with relu
        x = ReLU(name=self.block_name + '_relu')(x)

        return x


class ConvSoftmax(Block):
    """ ConvRele block for feature extraction:
        concatenate (if necessary, merge multiple input tensors)
        + convolutional layer
        + batch-normalization (if necessary)
        + soft-max

        the input of block can be one tensor or a list of tensors with the same shape
        the output of block is a tensor with the same shape of input
        the tensor is channel last
    """

    def __init__(self, ndim: int, k_num: int, k_size: int = 3, use_bn: bool = False, block_name: str = 'ConvSoftmax2D'):
        """
            :param ndim: the number of dimension of input, should be 2 or 3
            :param k_num: the number kernel for convolutional layer
            :param k_size: the size of kernel for convolutional layer
            :param use_bn: flag of batch normalization for convolutional layer
            :param block_name: block name
        """

        assert ndim in [2, 3]

        super().__init__(block_name)
        self.ndim = ndim
        self.k_num = k_num
        self.k_size = k_size
        self.use_bn = use_bn

    def __call__(self, x):
        # if input multiple tensors, concatenate them, channel last
        if isinstance(x, list):
            x = Concatenate(axis=-1, name=self.block_name + '_cate')(x)
        # convolutional
        if self.ndim == 2:
            x = Conv2D(filters=self.k_num, kernel_size=self.k_size, strides=1, padding='same',
                       use_bias=(not self.use_bn),
                       name=self.block_name + '_conv')(x)
        else:
            x = Conv3D(filters=self.k_num, kernel_size=self.k_size, strides=1, padding='same',
                       use_bias=(not self.use_bn),
                       name=self.block_name + '_conv')(x)
        # batch normalization
        if self.use_bn:
            x = BatchNormalization(axis=-1, name=self.block_name + '_bn')(x)
        # activation with relu
        x = Softmax(name=self.block_name + '_softmax')(x)

        return x


class DownSamplePooling(Block):
    """ The down sample using max-pooling:
        conv_block * b_num
        + max-pooling
        the input of block is a tensor or a list of tensors with the same shape
        the output of block is a tensor with the same shape of input, and a down sampled tensor.
    """

    def __init__(self, ndim: int, conv_block, b_num: int, down_size: int = 2, block_name: str = 'DownSample'):
        """
            :param ndim: the number of dimension of input, should be 2 or 3
            :param conv_block: the convolutional block for feature extraction, can be a block or layer
            :param b_num: the number of conv blocks
            :param down_size: the ratio of down_sample
            :param block_name: block name
        """

        assert ndim in [2, 3]
        assert ndim == conv_block.ndim

        super().__init__(block_name)
        self.ndim = ndim
        self.conv_block = conv_block
        self.b_num = b_num
        self.down_size = down_size

    def __call__(self, x):
        # conv_block
        for n in range(self.b_num):
            self.conv_block.rename(self.block_name + '.Conv_{}'.format(n))
            x = self.conv_block(x)

        # down sample with max pooling
        if self.ndim == 2:
            down = MaxPooling2D(pool_size=self.down_size, name=self.block_name + '_down')(x)
        else:
            down = MaxPooling3D(pool_size=self.down_size, name=self.block_name + '_down')(x)

        # return a list of tensors
        # the 1st is the down sample result, the 2nd is the feature maps from ConvRelu
        return [down, x]


class DownSampleConv(Block):
    """ The down sample using convolution:
        conv_block * b_num
        + convolution
        the input of block is a tensor or a list of tensors with the same shape
        the output of block is a tensor with the same shape of input, and a down sampled tensor.
    """

    def __init__(self, ndim: int, conv_block, b_num: int, down_size: int = 2, block_name: str = 'DownSample'):
        """
            :param ndim: the number of dimension of input, should be 2 or 3
            :param conv_block: the convolutional block for feature extraction, can be a block or layer
            :param b_num: the number of conv blocks
            :param down_size: the ratio of down_sample
            :param block_name: block name
        """

        assert ndim in [2, 3]
        assert ndim == conv_block.ndim

        super().__init__(block_name)
        self.ndim = ndim
        self.conv_block = conv_block
        self.b_num = b_num
        self.down_size = down_size

    def __call__(self, x):
        # conv_block
        for n in range(self.b_num):
            self.conv_block.rename(self.block_name + '.Conv_{}'.format(n))
            x = self.conv_block(x)

        # down sample with max pooling
        if self.ndim == 2:
            down = Conv2D(filters=x.shape[-1], kernel_size=3, strides=self.down_size, padding='same',
                          name=self.block_name + '_down')(x)
        else:
            down = Conv3D(filters=x.shape[-1], kernel_size=3, strides=self.down_size, padding='same',
                          name=self.block_name + '_down')(x)

        # return a list of tensors
        # the 1st is the down sample result, the 2nd is the feature maps from ConvRelu
        return [down, x]


class UpSample(Block):
    """ The up sample block using de-conv:
        up-sampling
        + conv_block * b_num
        the input of block is a tensor or a list of tensors (only the first tensor to be up-sampled)
        the output of block is a tensor.
    """

    def __init__(self, ndim: int, conv_block, b_num: int, up_size: int = 2, block_name: str = 'UpSample'):
        """
            :param ndim: the number of dimension of input, should be 2 or 3
            :param conv_block: the convolutional block for feature extraction, can be a block or layer
            :param b_num: the number of conv blocks
            :param up_size: he size of up-sampling
            :param block_name: block name
        """

        assert ndim in [2, 3]
        assert ndim == conv_block.ndim

        super().__init__(block_name)
        self.ndim = ndim
        self.conv_block = conv_block
        self.b_num = b_num
        self.up_size = up_size

    def __call__(self, x):
        # up sample
        if isinstance(x, list):
            if self.ndim == 2:
                x[0] = UpSampling2D(size=self.up_size, name=self.block_name + '_up')(x[0])
                x[0] = Conv2D(filters=x[0].shape[-1], kernel_size=3, strides=1, padding='same',
                              name=self.block_name + '_up_conv')(x[0])
            else:
                x[0] = UpSampling3D(size=self.up_size, name=self.block_name + '_up')(x[0])
                x[0] = Conv3D(filters=x[0].shape[-1], kernel_size=3, strides=1, padding='same',
                              name=self.block_name + '_up_conv')(x[0])
        else:
            if self.ndim == 2:
                x = UpSampling2D(size=self.up_size, name=self.block_name + '_up')(x)
                x = Conv2D(filters=x.shape[-1], kernel_size=3, strides=1, padding='same',
                           name=self.block_name + '_up_conv')(x)
            else:
                x = UpSampling3D(size=self.up_size, name=self.block_name + '_up')(x)
                x = Conv3D(filters=x.shape[-1], kernel_size=3, strides=1, padding='same',
                           name=self.block_name + '_up_conv')(x)

        # conv_block
        for n in range(self.b_num):
            self.conv_block.rename(self.block_name + '.Conv_{}'.format(n))
            x = self.conv_block(x)

        return x


# Test
def main():
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    conv_block = ConvRele(ndim=2, k_num=32, use_bn=False, block_name='ConvRelu2D')
    down_block = DownSampleConv(ndim=2, conv_block=conv_block, b_num=3, down_size=2, block_name='DownSample')
    up_block = UpSample(ndim=2, conv_block=conv_block, b_num=1, up_size=2, block_name='UpSample')
    x_in = Input(shape=(128, 128, 3))
    x = conv_block(x_in)
    x = down_block(x)
    x = up_block(x)

    model = Model(inputs=x_in, outputs=x)
    model.summary()
    print(model.input_shape)
    print(model.output_shape)


if __name__ == '__main__':
    main()
