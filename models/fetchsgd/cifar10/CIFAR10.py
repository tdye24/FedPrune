import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        # todo tdye finetune Dropout hyperparameters
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Flatten()

        )

        self.shared_clf = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # 50x3x32x32
        x = self.shared_conv(x)
        x = self.shared_clf(x)
        return x


if __name__ == '__main__':
    import torch
    model = CIFAR10()
    _x = torch.rand((50, 3, 32, 32))
    _output = model(_x)
    print(f'{_x.shape}->{_output.shape}')
    print("Parameters in total {}".format(sum(x.numel() for x in model.parameters())))