import torch
import torch.nn as nn

from . import common
from .ResNetT import ResNetT


def build_model(args):
    return MSRTNet(args)

class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            common.default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)

class MSRTNet(nn.Module):
    def __init__(self, args):
        super(MSRTNet, self).__init__()

        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size
        self.num_heads = [8,4,2,1]

        self.n_scales = args.n_scales
        input_resolutions = [args.patch_size//4, args.patch_size//2, args.patch_size]

        self.body_models = nn.ModuleList([
            ResNetT(args, 3, 3, mean_shift=False, input_resolution=input_resolutions[0], num_head=self.num_heads[0]),
        ])
        for i in range(1, self.n_scales):
            self.body_models.insert(0, ResNetT(args, 6, 3, mean_shift=False,input_resolution=input_resolutions[i],num_head=self.num_heads[i]))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]

    def forward(self, input_pyramid):

        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s, H=input_pyramid[s].shape[2], W=input_pyramid[s].shape[3])
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                # print("input_pyramid[s-1] shape:", input_pyramid[s-1].shape, "up_feat shape:", up_feat.shape)
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid
