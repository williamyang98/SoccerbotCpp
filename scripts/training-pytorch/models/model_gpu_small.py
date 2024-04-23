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
class SoccerBotModel_Gpu_Small(nn.Module):
    DOWNSCALE_RATIO = 1
    BATCH_SIZE = 64 
    STEPS_PER_EPOCH = 40

    def __init__(self):
        super().__init__()
        self.resize = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(7,7), groups=3, stride=(4,4), bias=True, padding=0),
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3,3), bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            SeparableConv2d(16, 32, (3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            SeparableConv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
        )
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        # x.shape = B,H,W,C where C=3
        # need to convert to B,C,H,W format since original application uses B,H,W,C and pytorch uses B,C,H,W
        y = x.permute(0,3,1,2)
        y = self.resize(y)
        y = self.feature_extractor(y)
        y = self.output_head(y)
        return y
