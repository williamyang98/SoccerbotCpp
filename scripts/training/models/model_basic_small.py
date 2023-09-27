import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Add
from tensorflow.keras.layers import LeakyReLU, ReLU

class SoccerBotModel_Basic_Small(tf.Module):
    DOWNSCALE_RATIO = 4

    def __init__(self):
        super().__init__()
        dropout_rate = 0.1
        lr_rate = 0.1
        
        self.conv_0 = Conv2D(16, (3,3))
        self.act_0 = LeakyReLU(lr_rate)
        self.pool_0 = MaxPooling2D((2,2))

        self.conv_1 = SeparableConv2D(32, (3,3))
        self.act_1 = LeakyReLU(lr_rate)
        self.pool_1 = MaxPooling2D((2,2))

        self.conv_2 = Conv2D(32, (3,3))
        self.act_2 = LeakyReLU(lr_rate)
        self.pool_2 = MaxPooling2D((2,2))

        self.conv_3 = SeparableConv2D(64, (3,3))
        self.act_3 = LeakyReLU(lr_rate)
        self.pool_3 = MaxPooling2D((3,3))

        self.flatten_5 = Flatten()
        self.dense_5 = Dense(64)
        self.act_5 = LeakyReLU(lr_rate)

        self.dense_6 = Dense(32)
        self.act_6 = LeakyReLU(lr_rate)

        self.dense_7 = Dense(3)

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

        y = self.flatten_5(y)
        y = self.dense_5(y)
        y = self.act_5(y)

        y = self.dense_6(y)
        y = self.act_6(y)

        y = self.dense_7(y)
        
        return y