import torch.nn as nn

from . import common
from .TBlock import LeWinTransformerBlock, InputProj, OutputProj

def build_model(args):
    return ResNet(args)

def get_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2*padding) // stride + 1

class ResNetT(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3, n_feats=None, kernel_size=None, n_resblocks=None, mean_shift=True, input_resolution=256, num_head=1):
        super(ResNetT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = args.n_feats if n_feats is None else n_feats
        self.kernel_size = args.kernel_size if kernel_size is None else kernel_size
        self.n_resblocks = args.n_resblocks if n_resblocks is None else n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2
        
        self.input_proj = InputProj()
        self.output_proj = OutputProj()

        modules = []
        modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
        res = get_conv_output_size(input_resolution, self.kernel_size, 1,(self.kernel_size // 2))
        for _ in range(self.n_resblocks):
            modules.append(common.ResBlock(self.n_feats, self.kernel_size))
        modules.append(self.input_proj)
        modules.append(LeWinTransformerBlock(dim=self.n_feats, input_resolution=(res,res), num_heads=num_head))
        # modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))
        
        self.body = nn.Sequential(*modules)
        self.end_conv = common.default_conv(self.n_feats, self.out_channels, self.kernel_size)

    def forward(self, input, H=None, W=None):
        if self.mean_shift:
            input = input - self.mean
        # print("H:",H," W:",W)

        output = self.body(input)
        output = self.output_proj(output,H,W)
        output = self.end_conv(output)

        if self.mean_shift:
            output = output + self.mean

        return output

