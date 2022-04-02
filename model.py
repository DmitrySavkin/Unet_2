import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv64 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv512 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.upconv1024 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dconv1024 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv512 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dconv512 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv256 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dconv256 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv128 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dconv128 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 183, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = F.relu(self.conv64(x / 255.))
        x2 = F.relu(self.conv128(self.pool(x1)))
        x3 = F.relu(self.conv256(self.pool(x2)))
        x4 = F.relu(self.conv512(self.pool(x3)))
        x5 = F.relu(self.conv1024(self.pool(x4)))
        ux5 = self.upconv1024(x5)
        cc5 = torch.cat([ux5, x4], 1)
        dx4 = F.relu(self.dconv1024(cc5))
        ux4 = self.upconv512(dx4)
        cc4 = torch.cat([ux4, x3], 1)
        dx3 = F.relu(self.dconv512(cc4))
        ux3 = self.upconv256(dx3)
        cc3 = torch.cat([ux3, x2], 1)
        dx2 = F.relu(self.dconv256(cc3))
        ux2 = self.upconv128(dx2)
        cc2 = torch.cat([ux2, x1], 1)
        dx1 = F.relu(self.dconv128(cc2))  # no relu?
        last = self.conv1(dx1)
        return last  # sigmoid if classes arent mutually exclusv

