import functools

import kornia
import torch.nn.functional as F

import models.archs.arch_util as arch_util
from models.archs.SFBlock import *


class FourLLIE(nn.Module):
    def __init__(self, nf=32):
        super(FourLLIE, self).__init__()

        # AMPLITUDE ENHANCEMENT
        self.AmpNet = nn.Sequential(AmplitudeNet_skip(8), nn.Sigmoid())

        self.nf = nf
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2d(3 * 2, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.spatial_frequency = SFNet(nf)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

        # TODO 在phase上整一整 不行
        # self.pha_conv = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
        # )

    @staticmethod
    def get_mask(dark):
        """算 SNR map"""
        light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1, 1))  # TODO: 1.5 -> 1
        dark = (
                dark[:, 0:1, :, :] * 0.299
                + dark[:, 1:2, :, :] * 0.587
                + dark[:, 2:3, :, :] * 0.114
        )  # RGB to grayscale
        light = (
                light[:, 0:1, :, :] * 0.299
                + light[:, 1:2, :, :] * 0.587
                + light[:, 2:3, :, :] * 0.114
        )
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        return mask.float()

    def forward(self, x):
        # AMPLITUDE ENHANCEMENT
        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm="backward")
        amp = torch.abs(image_fft)
        pha = torch.angle(image_fft)
        curve_amps = self.AmpNet(x)
        amp = amp / (curve_amps + 0.00000001)  # * d4
        # pha = self.pha_conv(pha)
        img_amp_enhanced = torch.fft.ifft2(
            torch.polar(amp, pha),
            s=(H, W),
            norm="backward",
        ).real
        x_center = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        # encoder
        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center, x), dim=1)))  # cat?
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        fea = self.feature_extraction(L1_fea_3)

        # 两路
        fea_light = self.recon_trunk_light(fea)  # F_spatial
        fea_unfold = self.spatial_frequency(fea)  # F_fourier

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = self.get_mask(x_center)
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='bilinear')  # TODO: nearest -> bilinear ; 普遍很小？

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)

        fea = fea_unfold * (1 - mask) + fea_light * mask  # F'

        # decoder
        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]

        return out_noise, amp, x_center, mask
