from torch import nn

class SElayer(nn.module):
    def __init__(self, channel, reduction = 16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        :wq!

