import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.framework.core.registry import MODELS
from src.framework.core.interfaces import Predictable


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, hidden=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class GatedSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.uncertainty_head = UncertaintyHead(in_channels)

    def forward(self, encoder_feat):
        u = self.uncertainty_head(encoder_feat)
        u_norm = torch.sigmoid(u)
        gate = 1.0 - u_norm
        return encoder_feat * gate, u_norm


class DecoderBlock(nn.Module):
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


@MODELS.register("faupnet")
class FAUPNet(nn.Module, Predictable):
    def __init__(self, in_channels=1, out_channels=1, encoder_name="mobilenet_v2",
                 gate_levels=None, pretrained=True):
        super().__init__()
        if gate_levels is None:
            gate_levels = [3, 4]
        self.gate_levels = gate_levels

        encoder = getattr(torchvision.models, encoder_name)(weights='IMAGENET1K_V1' if pretrained else None)
        features = encoder.features

        old_conv = features[0][0]
        old_bn = features[0][1]
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

        self.enc_1 = features[1:3]
        self.enc_2 = features[3:5]
        self.enc_3 = features[5:8]
        self.enc_4 = features[8:15]
        self.enc_5 = features[15:]

        self.encoder_channels = [32, 24, 32, 64, 160, 1280]
        dec_channels = [256, 128, 64, 32, 16]

        self.gates = nn.ModuleDict()
        for level in gate_levels:
            ec = self.encoder_channels[level]
            self.gates[str(level)] = GatedSkipConnection(ec)

        rev_enc = list(reversed(self.encoder_channels))
        self.decoder_blocks = nn.ModuleList()
        for i, dc in enumerate(dec_channels):
            skip_ch = rev_enc[i + 1] if i + 1 < len(rev_enc) else 0
            in_ch = rev_enc[i] if i == 0 else dec_channels[i - 1]
            in_ch = in_ch + skip_ch
            self.decoder_blocks.append(DecoderBlock(in_ch, dc))

        self.final = nn.Conv2d(dec_channels[-1], out_channels, 1)
        self.uncertainty_maps = {}

    def forward(self, x):
        self.uncertainty_maps = {}
        f0 = self.enc_0(x)
        f1 = self.enc_1(f0)
        f2 = self.enc_2(f1)
        f3 = self.enc_3(f2)
        f4 = self.enc_4(f3)
        f5 = self.enc_5(f4)
        features = [f0, f1, f2, f3, f4, f5]

        rev = features[::-1]
        dec = rev[0]
        for i, block in enumerate(self.decoder_blocks):
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=False)
            enc_idx = len(features) - 2 - i
            skip = features[enc_idx]
            gate_key = str(enc_idx)
            if gate_key in self.gates:
                skip, u = self.gates[gate_key](skip)
                self.uncertainty_maps[f"gate_level_{enc_idx}"] = u
            if dec.shape[2:] != skip.shape[2:]:
                dec = F.interpolate(dec, size=skip.shape[2:], mode='bilinear', align_corners=False)
            dec = torch.cat([dec, skip], dim=1)
            dec = block(dec)

        dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final(dec)

    def predict(self, x):
        return torch.sigmoid(self(x))

    def predict_with_uncertainty(self, x):
        y = self(x)
        return {"prediction": torch.sigmoid(y), "uncertainty_maps": self.uncertainty_maps}

    def get_uncertainty_maps(self):
        return self.uncertainty_maps
