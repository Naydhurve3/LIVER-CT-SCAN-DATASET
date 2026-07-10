import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.framework.core.registry import MODELS
from src.framework.core.interfaces import Predictable


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        rev_enc = encoder_channels[::-1]
        self.blocks = nn.ModuleList()
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = rev_enc[i + 1] if i + 1 < len(rev_enc) else 0
            in_ch = rev_enc[i] if i == 0 else decoder_channels[i - 1]
            self.blocks.append(ConvBlock(in_ch + skip_ch, out_ch))

    def forward(self, features):
        rev = features[::-1]
        x = rev[0]
        for i, block in enumerate(self.blocks):
            x = self.upsample(x)
            skip = rev[i + 1] if i + 1 < len(rev) else None
            if skip is not None:
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


@MODELS.register("mobilenetv2_unet")
class MobileNetV2UNet(nn.Module, Predictable):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        encoder = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)

        old_conv = encoder.features[0][0]
        old_bn = encoder.features[0][1]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.enc_0 = nn.Sequential(
            new_conv,
            nn.BatchNorm2d(old_bn.num_features, eps=old_bn.eps, momentum=old_bn.momentum),
            nn.ReLU(inplace=True),
        )
        if pretrained:
            with torch.no_grad():
                self.enc_0[1].weight.copy_(old_bn.weight)
                self.enc_0[1].bias.copy_(old_bn.bias)
                self.enc_0[1].running_mean.copy_(old_bn.running_mean)
                self.enc_0[1].running_var.copy_(old_bn.running_var)

        self.enc_1 = encoder.features[1:3]
        self.enc_2 = encoder.features[3:5]
        self.enc_3 = encoder.features[5:8]
        self.enc_4 = encoder.features[8:15]
        self.enc_5 = encoder.features[15:]

        self.decoder = UNetDecoder(
            encoder_channels=[32, 24, 32, 64, 160, 1280],
            decoder_channels=[256, 128, 64, 32, 16],
        )
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        f0 = self.enc_0(x)
        f1 = self.enc_1(f0)
        f2 = self.enc_2(f1)
        f3 = self.enc_3(f2)
        f4 = self.enc_4(f3)
        f5 = self.enc_5(f4)
        features = [f0, f1, f2, f3, f4, f5]
        dec = self.decoder(features)
        dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final(dec)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

    def predict_with_uncertainty(self, x):
        return {"prediction": torch.sigmoid(self.forward(x))}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
