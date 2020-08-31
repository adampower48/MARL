import torch
from torch.nn.modules import Module
from torch.nn.modules.activation import Softmax, ReLU
from torch.nn.modules.conv import Conv2d


class SimpleCNN(Module):
    can_train = True

    def __init__(self, input_channels, filter_size=3, num_filters=8):
        super().__init__()
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_filters = num_filters

        self._init_layers()

    def _init_layers(self):
        padding = self.filter_size // 2
        self.conv1 = Conv2d(self.input_channels, self.num_filters, kernel_size=self.filter_size, padding=padding)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(self.num_filters, self.num_filters, kernel_size=self.filter_size, padding=padding)
        self.relu2 = ReLU()
        self.conv3 = Conv2d(self.num_filters, 1, kernel_size=self.filter_size, padding=padding)
        self.softmax = Softmax(dim=0)

    def forward(self, x):
        img_shape = x.shape[-2:]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.softmax(x.view(-1)).view((-1, *img_shape))
        return x


class RandomCNN(Module):
    can_train = False

    def __init__(self):
        super().__init__()

        self.softmax = Softmax(dim=0)

    def forward(self, x):
        img_shape = x.shape[-2:]

        x = torch.rand((1, *img_shape))
        x = self.softmax(x.view(-1)).view((-1, *img_shape))

        return x
