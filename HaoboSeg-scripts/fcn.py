import torch.nn as nn


class fcn8s_old(nn.Module):
    def __init__(self, in_ch=3, n_class=3):
        super().__init__()
        # /4 downsampling
        self.down_1 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            nn.Conv2d(in_ch, 96, 3, padding='same', bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        # /8
        self.down_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(96, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.outconv_2 = nn.Sequential(
            nn.Conv2d(256, n_class, 1),
            nn.Softmax2d()
        )
        # /16
        self.down_3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.outconv_3 = nn.Sequential(
            nn.Conv2d(256, n_class, 1, padding='same'),
            nn.Softmax2d()
        )
        # /32
        self.down_4 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 4096, 3, padding='same', bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Conv2d(4096, 4096, 3, padding='same', bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Conv2d(4096, n_class, 1, padding='same'),
            # nn.Softmax2d()
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1, bias=False),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 16, 8, padding=4, bias=False),
            nn.Softmax2d()
        )

    def forward(self, x):
        x_down1 = self.down_1(x)                         # 1/4
        x_down2 = self.down_2(x_down1)                   # 1/8
        x_map2  = self.outconv_2(x_down2)
        x_down3 = self.down_3(x_down2)                   # 1/16
        x_map3  = self.outconv_3(x_down3)
        x_down4 = self.down_4(x_down3)                   # 1/32
        x_up3   = self.upconv(x_down4) + x_map3          # 1/16
        x_up2   = self.upconv(x_up3) + x_map2            # 1/8
        out     = self.upsample(x_up2)                   # 1
        return out


class fcn2s(nn.Module):
    def __init__(self, in_ch=3, n_class=3):
        super().__init__()
        # /2 downsampling
        self.down_1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_ch, 96, 3, padding='same', bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        
        self.outconv_1 = nn.Sequential(
            nn.Conv2d(96, n_class, 1),
            nn.ReLU()
        )
        # /4
        self.down_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(96, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.outconv_2 = nn.Sequential(
            nn.Conv2d(256, n_class, 1),
            nn.ReLU()
        )
        # /8
        self.down_3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 3, 3, padding='same', bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1, bias=False),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1, bias=False),
            nn.Softmax2d()
        )

    def forward(self, x):
        x_down1 = self.down_1(x)                         # 1/2
        x_map1  = self.outconv_1(x_down1)
        x_down2 = self.down_2(x_down1)                   # 1/4
        x_map2  = self.outconv_2(x_down2)
        x_down3 = self.down_3(x_down2)                   # 1/8       
        x_up2   = self.upconv(x_down3) + x_map2         # 1/4
        x_up1   = self.upconv(x_up2) + x_map1            # 1/2
        out     = self.upsample(x_up1)                   # 1
        return out


class fcn4s(nn.Module):
    def __init__(self, in_ch=3, n_class=3):
        super().__init__()
        # /2 downsampling
        self.down_1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_ch, 96, 3, padding='same', bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        
        self.outconv_1 = nn.Sequential(
            nn.Conv2d(96, n_class, 1),
            nn.Softmax2d()
        )
        # /4
        self.down_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(96, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.outconv_2 = nn.Sequential(
            nn.Conv2d(256, n_class, 1),
            nn.Softmax2d()
        )
        # /8
        self.down_3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding='same', bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 3, 3, padding='same', bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1, bias=False),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, padding=1, bias=False),
            nn.Softmax2d()
        )

    def forward(self, x):
        x_down1 = self.down_1(x)                         # 1/2
        x_map1  = self.outconv_1(x_down1)
        x_down2 = self.down_2(x_down1)                   # 1/4
        x_map2  = self.outconv_2(x_down2)
        x_down3 = self.down_3(x_down2)                   # 1/8       
        x_up2   = self.upconv(x_down3) + x_map2         # 1/4
        x_up1   = self.upconv(x_up2) + x_map1            # 1/2
        out     = self.upsample(x_up1)                   # 1
        return out

