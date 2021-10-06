import numpy as np

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')


class SimCLRTrainDataTransform(object):

    def __init__(
        self, input_size: tuple = (63,412), gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `transforms` from `torchvision` which is not installed yet.')

        self.jitter_strength = jitter_strength
        self.input_size = input_size
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            # transforms.RandomResizedCrop(size=self.input_size),
            transforms.RandomCrop(input_size, padding=(4, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p=0.2
            ),
            transforms.Grayscale(num_output_channels=1)
        ]

        '''if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))'''

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:

            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            transforms.RandomCrop(self.input_size, padding=(4, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),  # xys， 将输出图形变为单通道
            self.final_transform
        ])

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)

