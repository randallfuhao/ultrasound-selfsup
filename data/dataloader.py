import torch
import numpy as np
import skimage
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)  # reproducible


def compute_mean_std(datasfile=''):
    datas = np.load(datasfile)
    mean, std = [], []
    '''for i in range(1):
        channel = (datas[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        std.append(np.std(channel))'''

    grayimg = datas[:,:,:].ravel()/127.5 - 1
    grayimg = datas[:, :, :].ravel() / 255
    mean.append(np.mean(grayimg))
    std.append(np.std(grayimg))
    print('mean: ', mean)
    print('std: ', std)
    return mean, std



class MyDataset(Dataset):

    def __init__(self, datas, labels, transform, phase='train', channels=1):
        self.datas = np.load(datas)
        self.transforms = transform
        self.labels = np.load(labels)
        self.phase = phase
        self.channels = channels

    def __getitem__(self, index):
        example= self.datas[index, :, :]
        example = np.squeeze(example)
        if self.channels == 3:
            # example = np.expand_dims(example,2)
            # example = example.repeat(3, 2)
            example = np.expand_dims(example, 2).repeat(3, 2)

        example = Image.fromarray(np.uint8(example))
        example = self.transforms(example)
        label = self.labels[index]

        if self.phase == 'selfsup':
            return example
        else:
            return example, label
    def __len__(self):
        return self.datas.shape[0]


def build_dataloader_mulspk(datasfile, labelsfile, batchsize, phase='train', channels=1):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop((63,412), padding=(4,16)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    dataset = MyDataset(datasfile, labelsfile, transform, phase, channels)
    # loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=0)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=4)
    return loader


def build_dataloader_folder(dataFolder, batchsize, phase='train', channels=1, num_workers=0):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop((63,412), padding=(4,16)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

        dataset = torchvision.datasets.ImageFolder(dataFolder, transform)
        # loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, drop_last=True ,num_workers=num_workers)
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, drop_last=False,
                            num_workers=num_workers)
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

        dataset = torchvision.datasets.ImageFolder(dataFolder, transform)
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, drop_last=False, num_workers=num_workers)
        # loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=4)


    return loader


if __name__ == '__main__':
    '''datasfile = '../ultrasuite/01M_X.npy'
    labelsfile = '../ultrasuite/01M_label.npy'
    build_dataloader(datasfile,  labelsfile)'''
    compute_mean_std('/home/xys/pythonprj/UTI/ultrasuite/all/all_Xtrain.npy')
