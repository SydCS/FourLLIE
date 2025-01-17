import torch
import torch.nn as nn


class SpaBlock(nn.Module):
    """
    Spatial Processing (SP) Block
    """

    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class FreBlock(nn.Module):
    """
    Fourier Processing (FP) Block
    """

    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )
        # self.process1 = nn.Sequential(
        #     # Depthwise Convolution
        #     nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, groups=nc),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # Pointwise Convolution
        #     nn.Conv2d(nc, nc, kernel_size=1, stride=1, padding=0),
        # )
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0),
        )
        # self.process2 = nn.Sequential(
        #     nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, groups=nc),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(nc, nc, kernel_size=1, stride=1, padding=0),
        # )
        # TODO + spatial interaction 好像用处不大

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm="backward")
        amp = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        amp = self.process1(amp)
        pha = self.process2(pha)
        x_out = torch.fft.irfft2(torch.polar(amp, pha), s=(H, W), norm="backward")
        return x_out + x


class ProcessBlock(nn.Module):
    def __init__(self, nc, spatial=True):
        super(ProcessBlock, self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(nc)
        self.cat = (
            nn.Conv2d(2 * nc, nc, 1, 1, 0)
            if spatial
            else nn.Conv2d(nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        x_ori = x
        x_spatial = self.spatial_process(x)
        x_freq = self.frequency_process(x)
        x_cat = torch.cat([x_spatial, x_freq], 1)
        x_out = self.cat(x_cat) if self.spatial else self.cat(x_freq)
        return x_out + x_ori


class SFNet(nn.Module):
    """ F_fourier 支路 """

    def __init__(self, nc, n=1):
        super(SFNet, self).__init__()

        self.conv1 = ProcessBlock(nc, spatial=False)
        self.conv2 = ProcessBlock(nc, spatial=False)
        self.conv3 = ProcessBlock(nc, spatial=False)
        self.conv4 = ProcessBlock(nc, spatial=False)
        self.conv5 = ProcessBlock(nc, spatial=False)

    def forward(self, x):
        x_ori = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x_out = x_ori + x5
        return x_out


class AmplitudeNet_skip(nn.Module):
    """
    frequency stage: 估计 amplitude transform map
    """

    def __init__(self, nc, n=1):
        super(AmplitudeNet_skip, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            ProcessBlock(nc),
        )
        self.conv1 = ProcessBlock(nc)
        self.conv2 = ProcessBlock(nc)
        self.conv3 = ProcessBlock(nc)
        self.conv4 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, 3, 1, 1, 0),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout(torch.cat((x0, x5), dim=1))
        return xout
