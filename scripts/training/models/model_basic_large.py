import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Add
from tensorflow.keras.layers import LeakyReLU, ReLU

class SoccerBotModel_Basic_Large(tf.Module):
    DOWNSCALE_RATIO = 1

    def __init__(self):
        super().__init__()
        dropout_rate = 0.1
        lr_rate = 0.1
        
        # NOTE: Dropout on convolution layers is equivalent to introducing Bernoulli noise
        self.conv_0 = SeparableConv2D(32, (3,3))
        self.act_0 = LeakyReLU(lr_rate)
        self.pool_0 = MaxPooling2D((2,2))

        self.conv_1 = SeparableConv2D(64, (3,3))
        self.act_1 = LeakyReLU(lr_rate)
        self.pool_1 = MaxPooling2D((3,3))

        self.conv_2 = SeparableConv2D(64, (3,3))
        self.act_2 = LeakyReLU(lr_rate)
        self.pool_2 = MaxPooling2D((3,3))

        self.conv_3 = SeparableConv2D(128, (3,3))
        self.act_3 = LeakyReLU(lr_rate)
        self.pool_3 = MaxPooling2D((3,3))

        self.conv_4 = SeparableConv2D(256, (3,3))
        self.act_4 = LeakyReLU(lr_rate)
        self.pool_4 = MaxPooling2D((2,2))

        self.flatten_5 = Flatten()
        self.dropout_5 = Dropout(dropout_rate)
        self.bnorm_5 = BatchNormalization()
        self.dense_5 = Dense(64)
        self.act_5 = LeakyReLU(lr_rate)

        self.dense_6 = Dense(32)
        self.act_6 = LeakyReLU(lr_rate)

        self.dense_7 = Dense(16)
        self.act_7 = LeakyReLU(lr_rate)

        self.dense_8 = Dense(3)

    def __call__(self, x):
        # x.shape = B,H,W,C where C=3
        y = x
        y = self.conv_0(y)
        y = self.act_0(y)
        y = self.pool_0(y)

        y = self.conv_1(y)
        y = self.act_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.act_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.act_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.act_4(y)
        y = self.pool_4(y)

        y = self.flatten_5(y)
        y = self.dropout_5(y)
        y = self.bnorm_5(y)
        y = self.dense_5(y)
        y = self.act_5(y)

        y = self.dense_6(y)
        y = self.act_6(y)

        y = self.dense_7(y)
        y = self.act_7(y)

        y = self.dense_8(y)
        
        return y
