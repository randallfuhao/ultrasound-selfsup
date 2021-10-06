import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, (7, 7), padding=3),    # output:63,412,16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3, 3), padding=1),   # output:63,412,16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),             # output:31,206,8

            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )


        self.head = nn.Sequential(
            nn.Linear(15 * 103 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        fea = self.backbone(x)
        # mpool = self.maxpool2(fea)
        out = self.head(fea)
        return out

class MyModel_no1Dbn(nn.Module):
    def __init__(self):
        super(MyModel_no1Dbn, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, (7, 7), padding=3),    # output:63,412,16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3, 3), padding=1),   # output:63,412,16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),             # output:31,206,8

            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )


        self.head = nn.Sequential(
            nn.Linear(15 * 103 * 32, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        fea = self.backbone(x)
        # mpool = self.maxpool2(fea)
        out = self.head(fea)
        return out

class MyModel_no2Dbn(nn.Module):
    def __init__(self):
        super(MyModel_no2Dbn, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, (7, 7), padding=3),    # output:63,412,16
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3, 3), padding=1),   # output:63,412,16
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),             # output:31,206,8

            nn.Conv2d(16, 16, (3, 3), padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(15 * 103 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        fea = self.backbone(x)
        # mpool = self.maxpool2(fea)
        out = self.head(fea)
        return out
