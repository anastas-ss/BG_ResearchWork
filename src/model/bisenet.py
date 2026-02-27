import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        attn = self.attn(feat)
        return feat * attn


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        attn = self.attn(feat)
        return feat + feat * attn


# -------------------------
# Backbone: ResNet18-ish
# -------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # /2
        x = self.maxpool(x)   # /4
        feat1 = self.layer1(x)  # /4
        feat2 = self.layer2(feat1)  # /8
        feat3 = self.layer3(feat2)  # /16
        feat4 = self.layer4(feat3)  # /32
        return feat2, feat3, feat4


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)

        feat32_arm = self.arm32(feat32)
        avg = F.adaptive_avg_pool2d(feat32, 1)
        avg = self.conv_avg(avg)

        feat32_sum = feat32_arm + avg
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode="bilinear", align_corners=False)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode="bilinear", align_corners=False)
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)   # /2
        x = self.conv2(x)   # /4
        x = self.conv3(x)   # /8
        x = self.conv_out(x)
        return x


class BiSeNet(nn.Module):
    """
    BiSeNet v1 (ResNet18 backbone) для face parsing.
    Совместим с большинством весов на 19 классов (CelebAMask-HQ).
    """
    def __init__(self, n_classes=19):
        super().__init__()
        self.n_classes = n_classes

        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)

        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        sp = self.sp(x)                  # /8
        feat8, feat16_up, feat32_up = self.cp(x)  # feat8: /8

        fused = self.ffm(sp, feat16_up)  # /8

        out = self.conv_out(fused)
        out16 = self.conv_out16(feat16_up)
        out32 = self.conv_out32(feat32_up)

        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        out16 = F.interpolate(out16, size=(H, W), mode="bilinear", align_corners=False)
        out32 = F.interpolate(out32, size=(H, W), mode="bilinear", align_corners=False)

        # многие веса ожидают, что model(x) возвращает (out, out16, out32)
        return out, out16, out32
