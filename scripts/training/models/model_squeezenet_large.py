import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Add, MaxPooling1D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import Concatenate

class FireModule(tf.Module):
    def __init__(self, s1x1, e1x1, e3x3):
        super().__init__()
        lr_value = 0.1

        self.squeeze_1x1 = Conv2D(s1x1, (1,1))
        self.act_s1x1 = LeakyReLU(lr_value)
        self.expand_1x1 = Conv2D(e1x1, (1,1))
        self.act_e1x1 = LeakyReLU(lr_value)
        self.expand_3x3 = Conv2D(e3x3, (3,3), padding='same')
        self.act_e3x3 = LeakyReLU(lr_value)
        self.concat = Concatenate()

    def __call__(self, x):
        y = x
        y = self.squeeze_1x1(y)
        y = self.act_s1x1(y)

        y0 = self.expand_1x1(y)
        y1 = self.expand_3x3(y)

        y0 = self.act_e1x1(y0)
        y1 = self.act_e3x3(y1)

        y = self.concat([y0,y1])
        return y


class SoccerBotModel_Squeezenet_Large(tf.Module):
    DOWNSCALE_RATIO = 1

    def __init__(self):
        super().__init__()
        dropout_rate = 0.1
        lr_value = 0.1

        self.conv_0 = Conv2D(64, (3,3))
        self.act_0 = LeakyReLU(lr_value)
        self.pool_0 = MaxPooling2D((2,2))

        self.conv_1 = FireModule(32, 32, 32)
        self.act_1 = LeakyReLU(lr_value)
        self.pool_1 = MaxPooling2D((2,2))

        self.conv_2 = FireModule(32, 32, 32) 
        self.act_2 = LeakyReLU(lr_value)
        self.pool_2 = MaxPooling2D((2,2))

        self.conv_3 = FireModule(32, 32, 32) 
        self.act_3 = LeakyReLU(lr_value)
        self.pool_3 = MaxPooling2D((4,4))

        self.conv_4 = FireModule(32, 32, 32) 
        self.act_4 = LeakyReLU(lr_value)
        self.pool_4 = MaxPooling2D((4,4))

        self.flatten_5 = Flatten()
        self.dropout_5 = Dropout(dropout_rate)
        self.bnorm_5 = BatchNormalization()
        self.dense_5 = Dense(64)
        self.act_5 = LeakyReLU(lr_value)

        self.dense_6 = Dense(32)
        self.act_6 = LeakyReLU(lr_value)

        self.dense_7 = Dense(16)
        self.act_7 = LeakyReLU(lr_value)

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
