import torch
import torch.nn as nn

from . import common
from .ResNet import ResNet


def build_model(args):
    return MSResNet(args)

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
    

class MSResNet(nn.Module):
    def __init__(self, args):
        super(MSResNet, self).__init__()

        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size

        self.n_scales = args.n_scales

        self.body_models = nn.ModuleList([
            ResNet(args, 3, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet(args, 6, 3, mean_shift=False))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]
        # print(self.n_scales)
        # 3
        # print(self.body_models)
        # ModuleList(
        #   (0): ResNet(
        #     (body): Sequential(
        #       (0): Conv2d(6, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #       (1): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (2): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (3): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (4): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (5): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (6): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (7): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (8): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (9): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (10): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (11): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (12): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (13): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (14): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (15): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (16): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (17): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (18): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (19): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (20): Conv2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #     )
        #   )
        #   (1): ResNet(
        #     (body): Sequential(
        #       (0): Conv2d(6, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #       (1): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (2): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (3): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (4): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (5): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (6): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (7): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (8): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (9): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (10): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (11): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (12): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (13): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (14): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (15): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (16): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (17): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (18): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (19): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (20): Conv2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #     )
        #   )
        #   (2): ResNet(
        #     (body): Sequential(
        #       (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #       (1): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (2): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (3): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (4): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (5): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (6): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (7): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (8): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (9): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (10): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (11): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (12): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (13): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (14): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (15): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (16): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (17): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (18): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (19): ResBlock(
        #         (body): Sequential(
        #           (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #           (1): ReLU(inplace=True)
        #           (2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #         )
        #       )
        #       (20): Conv2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #     )
        #   )
        # )
        # print(self.conv_end_models)
        # ModuleList(
        #   (0): None
        #   (1): conv_end(
        #     (uppath): Sequential(
        #       (0): Conv2d(3, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #       (1): PixelShuffle(upscale_factor=2)
        #     )
        #   )
        #   (2): conv_end(
        #     (uppath): Sequential(
        #       (0): Conv2d(3, 12, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        #       (1): PixelShuffle(upscale_factor=2)
        #     )
        #   )
        # )

    def forward(self, input_pyramid):

        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse
        print(input_pyramid[0].shape)
        print(input_pyramid[1].shape)
        print(input_pyramid[2].shape)

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean
            # print(output_pyramid[s].shape)
            # torch.Size([1, 3, 180, 320])
            # torch.Size([1, 3, 360, 640])
            # torch.Size([1, 3, 720, 1280])

        return output_pyramid
