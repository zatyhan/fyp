import os
import torch, torchvision, torchinfo
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision import transforms
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional
# from torch.hub import load_state_dict_from_url
import numpy as np
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
# torch.use_deterministic_algorithms(True)

# defines a factory function to produce the encoder
def seg_resnet34(attention=None, in_ch=3, pretrained=False):

    resnet34 = torchvision.models.resnet34(pretrained=pretrained)
    cnn = SegResNet(BasicBlock, [3, 4, 6, 3], attention=attention, in_ch=in_ch)
    #读取参数
    if pretrained:
        pretrained_dict = resnet34.state_dict()
        model_dict = cnn.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        cnn.load_state_dict(model_dict)
    return cnn


# the encoder class
class SegResNet(ResNet):

    def __init__(self, block, layers, attention='None', in_ch=3):
        super().__init__(block, layers)
        self.out_channels = (64, 64, 128, 256, 512)
        self.attention1 = Attention(attention, in_channels=64)
        self.attention2 = Attention(attention, in_channels=64)
        self.attention3 = Attention(attention, in_channels=128)
        self.attention4 = Attention(attention, in_channels=256)
        self.attention5 = Attention(attention, in_channels=512)
        self.in_ch = in_ch
        del self.fc
        del self.avgpool
        if not in_ch==3:
            del self.conv1
            self.conv1_6 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    def get_stages(self):
        if self.in_ch==6:
            return [
                nn.Identity(),
                nn.Sequential(self.conv1_6, self.bn1, self.relu),
                self.attention1,
                nn.Sequential(self.maxpool, self.layer1),
                self.attention2, 
                self.layer2,
                self.attention3, 
                self.layer3,
                self.attention4,
                self.layer4,
                self.attention5,
            ]
        else:
            return [
                nn.Identity(),
                nn.Sequential(self.conv1, self.bn1, self.relu),
                self.attention1,
                nn.Sequential(self.maxpool, self.layer1),
                self.attention2, 
                self.layer2,
                self.attention3, 
                self.layer3,
                self.attention4,
                self.layer4,
                self.attention5,
            ]

    # def load_state_dict(self, state_dict, **kwargs):
    #     state_dict.pop("fc.bias", None)
    #     state_dict.pop("fc.weight", None)
    #     super().load_state_dict(state_dict, **kwargs)
        
    def forward(self, x):
        features = []
        stages = self.get_stages()
        for layer in stages:
            x = layer(x)
            if isinstance(layer, Attention):
                
                features.append(x)
        return features



class SCSEModule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                # [N, C, H, W] -> [N, C, 1, 1]
            nn.Conv2d(in_channels, in_channels//reduction, 1),      # [N, C, 1, 1] -> [N, C/2, 1, 1]
            nn.ReLU(inplace=True),                             
            nn.Conv2d(in_channels//reduction, in_channels, 1),      # [N, C/2, 1, 1] -> [N, C, 1, 1]
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1),      # [N, C, H, W] -> [N, 1, H, W]
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)



class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)



### following classes supports the decoder
class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), Dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout2d(Dropout) if Dropout else nn.Identity()
        # self.encoder = seg_resnet34()
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding="same", 
                              bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.dropout(input)
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return(out)
        # features = self.encoder(input)
        # features  = features[::-1]



class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 kernel_size=(3, 3), 
                 attention=None,
                 Dropout = 0.2):

        super().__init__()
        # print(in_channels+skip_channels)
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest",)
        self.conv1 = Conv2dBnRelu(in_channels + skip_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    Dropout=Dropout
                                    )

        self.attention1 = Attention(attention, in_channels = in_channels + skip_channels)

        self.conv2 = Conv2dBnRelu(out_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size, 
                                    Dropout=Dropout
                                    )

        self.attention2 = Attention(attention, in_channels=out_channels)                                    
    
    def forward(self, x:torch.Tensor, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # print(skip.shape)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            attention_type="scse",
            Dropout = 0.2
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # print(encoder_channels)
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
       # print(in_channels, skip_channels, out_channels)
        
        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(Dropout=0.2, attention=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, Dropout=0.2, attention=attention_type)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features:List[torch.Tensor]):

        # features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # print(features)
        head = features[0]
        skips = features[1:]

        # print(len(skips))
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            # print("decoder_block")
            skip = skips[i] if i < len(skips) else None
            # print(x.shape, skip.shape)
            # print(skip.shape)
            x = decoder_block(x, skip)

        return x

class Unet(nn.Module):
    def __init__(
        self,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        attention=None, 
        # activation=nn.Softmax(dim=1),
        activation=nn.Sigmoid(),
        in_ch=3,
        classes=3,
        Dropout = 0.2,
        pretrained=False
    ):
        super().__init__()
        self.encoder = seg_resnet34(attention=attention, in_ch=in_ch)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels, 
            decoder_channels=decoder_channels, 
            attention_type=attention,
            Dropout = Dropout
        )
        self.dropout = nn.Dropout2d(Dropout) if Dropout else nn.Identity()
        self.outconv = nn.Conv2d(decoder_channels[-1], classes, 3, padding="same")
        self.activation = activation

    def forward(self, x):
        # print(self.encoder.out_channels)
        features = self.encoder(x)
        # print(len(features))
        decoder_out = self.decoder(features)
        seg_out = self.dropout(decoder_out)
        seg_out = self.outconv(decoder_out)

        if self.activation is not None:
            out = self.activation(seg_out)
        return out
