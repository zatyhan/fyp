import torch
import torch.nn as nn

class log_dice_loss(nn.Module):
    def __init__(self, eps = 1e-7, channels=3, channel_wise=True):
        super().__init__()
        self.eps = eps
        self.channel_wise = channel_wise
        self.channels = channels

    def forward(self, y_pred, y_true):
        if self.channel_wise:
            # for i in self.channels:
            nom = 2 * torch.sum(y_pred * y_true, dim=[-2, -1]) # [N, C, H, W] -> [N, C]
            # print(.shape)
            # nom = (nom.sum(dim=[-2, -1]) + self.eps) / self.channels
            denom = torch.sum(y_pred + y_true, dim=[-2, -1]) + self.eps
            frac = nom / denom
            frac = frac.sum(dim=1) # [N, C] -> [N]
            frac /= self.channels
            loss = (-torch.log(frac)) ** 0.3
            loss = loss.mean()   # [N] -> num
            # print(loss.shape)
            return loss
            
        else:
            pass

            
