import torch
import torch.nn as nn
from torch.autograd import Variable


class Gen(nn.Module):
    def __init__(self, channels):
        super(Gen, self).__init__()

        # Layers: 
        self.block1_1 = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.PReLU()
        )
        # Layers: 
        self.block1_2 = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 4, 0),
            nn.PReLU()
        )
        # Layers: 
        self.block1_3 = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 8, 8, 0),
            nn.PReLU()
        )

        # Layers: 
        self.block2_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.PReLU()
        )

        # Layers: 
        self.block2_2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.PReLU()
        )

        # Layers: 
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        )

        # Layers: 
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        )

        # Layers: 
        self.block5 = nn.Sequential(
            nn.Conv2d(96, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(64 + channels, 64, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
        )

    def forward(self, input):
        downsample1_1 = self.block1_1(input)
        downsample1_2 = self.block1_2(input)
        downsample1_3 = self.block1_3(input)

        downsample2 = self.block2_1(downsample1_1)
        concat_2 = torch.cat((downsample2, downsample1_2), dim=1)
        downsample3 = self.block2_2(concat_2)
        concat_3 = torch.cat((downsample3, downsample1_3), dim=1)
        upsample1 = self.block3(concat_3)
        concat1 = torch.cat((upsample1, downsample2), dim=1)
        upsample2 = self.block4(concat1)
        concat2 = torch.cat((upsample2, downsample1_1), dim=1)
        upsample3 = self.block5(concat2)
        concat3 = torch.cat((upsample3, input), dim=1)

        return self.block6(concat3)


class Dis(nn.Module):
    def __init__(self, channels):
        super(Dis, self).__init__()
        features = 32
        self.main = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(features, 2 * features, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * features, 4 * features, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * features, 4 * features, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * features, 1, 4, 1, padding=0),
        )

    def forward(self, input):
        output = self.main(input)
        out = torch.nn.Sigmoid()(output)
        return out.view(-1, 1)
