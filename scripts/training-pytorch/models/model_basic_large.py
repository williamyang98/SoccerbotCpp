import torch
import torch.nn as nn

# create model
class SoccerBotModel_Basic_Large(nn.Module):
    DOWNSCALE_RATIO = 1
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = 40
    
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(3, 16, (3,3))
        self.act_0 = nn.LeakyReLU()
        self.pool_0 = nn.MaxPool2d((2,2))

        self.conv_1 = nn.Conv2d(16, 32, (3,3))
        self.act_1 = nn.LeakyReLU()
        self.pool_1 = nn.MaxPool2d((2,2))

        self.conv_2 = nn.Conv2d(32, 64, (3,3))
        self.act_2 = nn.LeakyReLU()
        self.pool_2 = nn.MaxPool2d((2,2))

        self.conv_3 = nn.Conv2d(64, 128, (3,3))
        self.act_3 = nn.LeakyReLU()
        self.pool_3 = nn.MaxPool2d((2,2))

        self.conv_4 = nn.Conv2d(128, 256, (3,3))
        self.act_4 = nn.LeakyReLU()
        self.pool_4 = nn.MaxPool2d((2,2))

        self.conv_5 = nn.Conv2d(256, 128, (3,3))
        self.act_5 = nn.LeakyReLU()
        self.pool_5 = nn.MaxPool2d((2,2))

        self.flatten_6 = nn.Flatten()
        self.dense_6 = nn.Linear(1920, 128)
        self.act_6 = nn.LeakyReLU()

        self.dense_7 = nn.Linear(128, 64)
        self.act_7 = nn.LeakyReLU()

        self.dense_8 = nn.Linear(64, 32)
        self.act_8 = nn.LeakyReLU()

        self.dense_9 = nn.Linear(32, 3)

    def forward(self, x):
        # x.shape = B,H,W,C where C=3
        # need to convert to B,C,H,W format since original application uses B,H,W,C and pytorch uses B,C,H,W
        y = x.permute(0,3,1,2)
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

        y = self.conv_5(y)
        y = self.act_5(y)
        y = self.pool_5(y)

        y = self.flatten_6(y)
        y = self.dense_6(y)
        y = self.act_6(y)

        y = self.dense_7(y)
        y = self.act_7(y)

        y = self.dense_8(y)
        y = self.act_8(y)

        y = self.dense_9(y)
        
        return y
