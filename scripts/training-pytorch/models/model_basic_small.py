import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=0)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(y)
        return y

# create model
class SoccerBotModel_Basic_Small(nn.Module):
    DOWNSCALE_RATIO = 4
    BATCH_SIZE = 128
    STEPS_PER_EPOCH = 40
    
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(3, 16, (3,3))
        self.act_0 = nn.ReLU()
        self.pool_0 = nn.MaxPool2d((2,2))

        self.conv_1 = SeparableConv2d(16, 32, (3,3))
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d((2,2))

        self.conv_2 = nn.Conv2d(32, 32, (3,3))
        self.act_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d((2,2))

        self.conv_3 = SeparableConv2d(32, 64, (3,3))
        self.act_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d((3,3))

        self.flatten_5 = nn.Flatten()
        self.dense_5 = nn.Linear(384, 64)
        self.act_5 = nn.ReLU()

        self.dense_6 = nn.Linear(64, 32)
        self.act_6 = nn.ReLU()

        self.dense_7 = nn.Linear(32, 3)

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

        y = self.flatten_5(y)
        y = self.dense_5(y)
        y = self.act_5(y)

        y = self.dense_6(y)
        y = self.act_6(y)

        y = self.dense_7(y)
        
        return y