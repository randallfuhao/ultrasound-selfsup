import os
import torch
import numpy as np
import skimage
import os
from PIL import Image
from imageio import imread
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ustools.read_core_files import parse_parameter_file,read_ultrasound_file

# torch.manual_seed(1)  # reproducible



def saveUlt2imags(path_ult, img_savepath):
    path = path_ult if path_ult!='' else r"/media/xys/work/UltraSuite/core-uxtd/core"
    speaker_list = os.listdir(path)

    all_x = np.empty((0, 63, 412), int)
    all_label = np.array([], dtype=int)

    for speaker_id in range(len(speaker_list)):
        print(speaker_list[speaker_id], '-----------------------------', '\n')
        for f_name in os.listdir(os.path.join(path, speaker_list[speaker_id])):
            if f_name.endswith('.ult'):
                ## Read ultrasound files
                ult = read_ultrasound_file(os.path.join(path, speaker_list[speaker_id], f_name))
                ## Read parameters
                param_df = parse_parameter_file(os.path.join(path, speaker_list[speaker_id], f_name[:-4]) + '.param')
                ## Reshape ult data
                ult_3d = ult.reshape(-1, int(param_df['NumVectors'].value), int(param_df['PixPerVector'].value))
                for i in range(len(ult_3d)):
                    im = Image.fromarray(ult_3d[i])
                    savepath = img_savepath if img_savepath!='' else r'/media/xys/work/UltraSuite/selfsupImgs/uxtd_imgs'
                    im.save( os.path.join(savepath,  f'{speaker_list[speaker_id]}_{i}.jpg') )
                    pass

def saveUlt2imags_therapy(path_ult='', img_savepath='', prefix='upx'):

    path = path_ult if path_ult!='' else r"/media/xys/Elsa/UltraSuite/core-upx/core"
    speaker_list = os.listdir(path)

    all_x = np.empty((0, 63, 412), int)
    all_label = np.array([], dtype=int)

    for speaker_id in range(len(speaker_list)):
        print(speaker_list[speaker_id], '-----------------------------', '\n')
        savepath = img_savepath if img_savepath != '' else r'/media/xys/work/UltraSuite/selfsupImgs/'
        os.makedirs(os.path.join(savepath, f'{prefix}_{speaker_list[speaker_id]}'))
        for dir in os.listdir(os.path.join(path, speaker_list[speaker_id])):
            for f_name in os.listdir(os.path.join(path, speaker_list[speaker_id], dir)):
                if f_name.endswith('.ult'):
                    ## Read ultrasound files
                    ult = read_ultrasound_file(os.path.join(path, speaker_list[speaker_id], dir, f_name))
                    ## Read parameters
                    param_df = parse_parameter_file(os.path.join(path, speaker_list[speaker_id], dir, f_name[:-4]) + '.param')
                    ## Reshape ult data
                    ult_3d = ult.reshape(-1, int(param_df['NumVectors'].value), int(param_df['PixPerVector'].value))
                    for i in range(len(ult_3d)):
                        im = Image.fromarray(ult_3d[i])
                        im.save( os.path.join(savepath, f'{prefix}_{speaker_list[speaker_id]}', f'{dir}_{i}.jpg') )
                        pass


def compute_mean_std(datafolder=''):
    startt = 5000
    CNum = 10000
    R_channel = 0
    R_channel_square = 0
    pixels_num = 0

    files = os.listdir(datafolder)
    imgs = []
    for i in range(startt, startt + CNum):
        img = imread(os.path.join(datafolder, files[i]))
        h, w = img.shape
        pixels_num += h * w

        R_temp = img[:, :]/127.5-1
        R_temp = img[:, :] / 255
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))

    R_mean = R_channel / pixels_num

    R_std = np.sqrt(R_channel_square / pixels_num - R_mean * R_mean)

    print("mean is %f" % (R_mean))
    print("std is %f" % (R_std))
    return R_mean, R_std




def build_dataloader(dataFolder, batchsize, udeftransforms=None):
    '''
    :param datasfile:
    :param labelsfile:
    :param phase: train, test, selfsup
    :param batchsize:
    :param systemseed:
    :return:
    '''

    # mean, std = compute_mean_std(dataFolder)
    if udeftransforms == None:
        transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.RandomCrop((63,412), padding=(4,16)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    else:
        transform = udeftransforms

    dataset = torchvision.datasets.ImageFolder(dataFolder, transform)
    # loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=0)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=False, num_workers=4)
    return loader

if __name__ == '__main__':
    # saveUlt2imags(r"/media/xys/Elsa/Tal80/TaL80/core", r'/media/xys/work/UltraSuite/selfsupImgs/TaL80_imgs')
    # saveUlt2imags_therapy()
    saveUlt2imags_therapy(r"/media/xys/Elsa/UltraSuite/core-uxssd/core", r'/media/xys/work/UltraSuite/selfsupImgs_3U', prefix='uxssd')
    # compute_mean_std('/media/xys/work/UltraSuite/uxtd_imgs/selfsup')
