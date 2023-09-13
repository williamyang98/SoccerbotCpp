import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Add
from tensorflow.keras.layers import LeakyReLU, ReLU

class SoccerBotModel(tf.Module):
    def __init__(self):
        super().__init__()

        self.sep_conv2d_0 = Conv2D(16, (3,3))
        self.relu_0 = ReLU()
        self.maxpool_0 = MaxPooling2D((2,2))

        self.sep_conv2d_1 = SeparableConv2D(32, (3,3))
        self.relu_1 = ReLU()
        self.maxpool_1 = MaxPooling2D((2,2))

        self.sep_conv2d_2 = Conv2D(32, (3,3))
        self.relu_2 = ReLU()
        self.maxpool_2 = MaxPooling2D((2,2))

        self.sep_conv2d_3 = SeparableConv2D(64, (3,3))
        self.relu_3 = ReLU()
        self.maxpool_3 = MaxPooling2D((3,3))

        self.flatten_5 = Flatten()
        self.dense_5 = Dense(64)
        self.relu_5 = ReLU()

        self.dense_6 = Dense(32)
        self.relu_6 = ReLU()

        self.dense_7 = Dense(3)

    def __call__(self, x):
        # x.shape = B,H,W,C where C=3
        y = x
        y = self.sep_conv2d_0(y)
        y = self.relu_0(y)
        y = self.maxpool_0(y)

        y = self.sep_conv2d_1(y)
        y = self.relu_1(y)
        y = self.maxpool_1(y)

        y = self.sep_conv2d_2(y)
        y = self.relu_2(y)
        y = self.maxpool_2(y)

        y = self.sep_conv2d_3(y)
        y = self.relu_3(y)
        y = self.maxpool_3(y)

        y = self.flatten_5(y)
        y = self.dense_5(y)
        y = self.relu_5(y)

        y = self.dense_6(y)
        y = self.relu_6(y)

        y = self.dense_7(y)
        
        return y