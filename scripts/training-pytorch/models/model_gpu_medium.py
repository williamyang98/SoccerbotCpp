import torch
import torch.nn as nn

# create model
class SoccerBotModel_Gpu_Medium(nn.Module):
    DOWNSCALE_RATIO = 1
    BATCH_SIZE = 32 
    STEPS_PER_EPOCH = 40
 
    def __init__(self):
        super().__init__()

        self.resize = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(5,5), groups=3, stride=(2,2), bias=True, padding=0),
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, (3,3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (3,3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 64, (3,3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2)),
        )
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
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
