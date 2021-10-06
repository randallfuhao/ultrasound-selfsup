import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import shutil
import random
from ustools.read_core_files import parse_parameter_file,read_ultrasound_file
from ustools.transform_ultrasound import transform_ultrasound
from ustools.visualise_ultrasound import display_2d_ultrasound_frame
import matplotlib.pyplot as plt

Bilabial_Labiodental = [ 'b', 'p', 'm', 'f', 'v']
Dental_Alveolar = ['T', 'D', 't', 'd', 's', 'z', 'n', 'l', 'S', '3', 'tS', 'dZ']
Velar = ['k', 'g', 'N', 'w']
Approximant = ['r']

meta_infor = pd.read_csv(r"/media/xys/work/UltraSuite/all_speaker_info.csv")

phonelabel_list = []
for f_name in os.listdir(r"/media/xys/work/UltraSuite/labels-uxtd-uxssd-upx/uxtd/phone_labels/lab"):
    phonelabel_list.append(f_name)



def dataset_div(datasfile, labelsfile):
    '''
    :param datasfile:
    :param labelsfile:
    :return:
    '''

    np.random.seed(1)   # Ensure reproducibility of data set partitioning
    datas = np.load(datasfile)
    labels = np.load(labelsfile)
    index_shuf = list(range(len(datas)))
    np.random.shuffle(index_shuf)
    p1 = int(len(datas)*0.6)
    p2 = int(len(datas)*0.8)
    index_train = index_shuf[0:p1]
    index_valid = index_shuf[p1:p2]
    index_test  = index_shuf[p2:]
    datas_train = datas[index_train]
    labels_train = labels[index_train]
    datas_valid = datas[index_valid]
    labels_valid = labels[index_valid]
    datas_test = datas[index_test]
    labels_test = labels[index_test]

    np.save(datasfile[:-4]+'train.npy', np.array(datas_train))
    np.save(labelsfile[:-4]+'train.npy', np.array(labels_train))
    np.save(datasfile[:-4] + 'valid.npy', np.array(datas_valid))
    np.save(labelsfile[:-4] + 'valid.npy', np.array(labels_valid))
    np.save(datasfile[:-4] + 'test.npy', np.array(datas_test))
    np.save(labelsfile[:-4] + 'test.npy', np.array(labels_test))


def make_multispeaker():
    path = r"/media/xys/work/UltraSuite/core-uxtd/core"
    speaker_list = os.listdir(path)
    speaker_list.sort(key=lambda x: int(x[:2]))

    all_x = np.empty((0,63,412), int)
    all_label = np.array([], dtype=int)

    for speaker_id in range(len(speaker_list)):
        print(speaker_list[speaker_id], '-----------------------------', '\n')
        speaker_x = np.empty((0, 63, 412), int)
        speaker_label = np.array([], dtype=int)
        for f_name in os.listdir(os.path.join(path, speaker_list[speaker_id])):
            if f_name.endswith('.ult') and ( ('A' in f_name) or ('B' in f_name) ):
                ## Read ultrasound files
                ult = read_ultrasound_file(os.path.join(path, speaker_list[speaker_id], f_name))

                ## Read parameters
                param_df = parse_parameter_file(os.path.join(path, speaker_list[speaker_id], f_name[:-4]) + '.param')
                TimeInSecsOfFirstFrame = param_df['TimeInSecsOfFirstFrame'].value
                FramesPerSec = param_df['FramesPerSec'].value

                ## Reshape ult data
                ult_3d = ult.reshape(-1, int(param_df['NumVectors'].value), int(param_df['PixPerVector'].value))
                plt.imshow(ult_3d[0,:,:])
                plt.show()
                vis_frame1 = transform_ultrasound(ult_3d[0])
                display_2d_ultrasound_frame(vis_frame1[0], dpi=None, figsize=(10, 10))

                ## read phone label file
                phonelabel_file = '/media/xys/work/UltraSuite/labels-uxtd-uxssd-upx/uxtd/phone_labels/lab/{}-{}.lab'.format(speaker_list[speaker_id], f_name[:-4])
                pd_phone = pd.read_table(phonelabel_file, sep=' ', index_col=None, header=None, names=["begin", 'end', 'phonetype'])
                for row in pd_phone.itertuples():
                    phonetype = getattr(row, 'phonetype')
                    label = None
                    if phonetype in Bilabial_Labiodental:
                        label = 0
                    elif phonetype in Dental_Alveolar:
                        label = 1
                    elif phonetype in Velar:
                        label = 2
                    elif phonetype in Approximant:
                        label = 3
                    if label != None:
                        begin = getattr(row, 'begin')
                        end = getattr(row, 'end')
                        beginFrame = max( (begin/1e7 - TimeInSecsOfFirstFrame) * FramesPerSec, 1)
                        endframe =   max( (end/1e7 - TimeInSecsOfFirstFrame) * FramesPerSec, 1)
                        centerframe = int( (beginFrame + endframe)//2 )
                        temp = np.expand_dims( ult_3d[centerframe], 0)
                        speaker_x = np.concatenate( (speaker_x, temp), axis=0)
                        speaker_label =  np.concatenate((speaker_label, np.array(label, ndmin=1)), axis=0)

        np.save(r"/home/xys/pythonprj/UTI/ultrasuite/{}_X.npy".format(speaker_list[speaker_id]), np.array(speaker_x))
        np.save(r"/home/xys/pythonprj/UTI/ultrasuite/{}_label.npy".format(speaker_list[speaker_id]), np.array(speaker_label))
        all_x = np.concatenate((all_x, speaker_x), axis=0)
        all_label = np.concatenate((all_label, speaker_label), axis=0)

    np.save(r"/home/xys/pythonprj/UTI/ultrasuite/all/all_X.npy", np.array(all_x))
    np.save(r"/home/xys/pythonprj/UTI/ultrasuite/all/all_label.npy", np.array(all_label))

    dataset_div('/home/xys/pythonprj/UTI/ultrasuite/all/all_X.npy',
                '/home/xys/pythonprj/UTI/ultrasuite/all/all_label.npy', )


def dataset_div_folder(dataroot):
    trainRatio = 0.6
    validRatio = 0.2
    # dataroot_parent = os.path.dirname(dataroot)
    train_root = os.path.join(dataroot, 'train')
    valid_root = os.path.join(dataroot, 'valid')
    test_root = os.path.join(dataroot, 'test')

    np.random.seed(20)

    dir_categories = os.listdir(dataroot)
    for dir in dir_categories:
        cnt = 0
        imgs = os.listdir(os.path.join(dataroot, dir))
        for image in imgs:
            cnt = cnt + 1
        cnt_train = int(cnt * trainRatio)
        cnt_valid = int(cnt * validRatio)
        imgs_train = np.random.choice(imgs, cnt_train, replace=False)
        imgs_valtest = list(set(imgs) - set(imgs_train))
        imgs_valid = np.random.choice(imgs_valtest, cnt_valid, replace=False)
        imgs_test = list(set(imgs_valtest) - set(imgs_valid))
        # print(imgs_train)
        # print(imgs_test)

        # move  to dirs
        dst_dir = os.path.join(train_root, dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in imgs_train:
            shutil.copyfile(os.path.join(dataroot, dir, img), os.path.join(dst_dir, img))
        dst_dir = os.path.join(valid_root, dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in imgs_valid:
            shutil.copyfile(os.path.join(dataroot, dir, img), os.path.join(dst_dir, img))
        dst_dir = os.path.join(test_root, dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for img in imgs_test:
            shutil.copyfile(os.path.join(dataroot, dir, img), os.path.join(dst_dir, img))


def make_speakerDependent():
    path = r"/media/xys/work/UltraSuite/core-uxtd/core"
    savepath = r'/media/xys/work/UltraSuite/speakerDependentImgs/'
    speaker_list = os.listdir(path)
    speaker_list.sort(key=lambda x: int(x[:2]))

    for speaker_id in range(len(speaker_list)):
        for label in range(4):
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', f'{label}')
            os.makedirs(savedir, exist_ok=True)

    for speaker_id in range(len(speaker_list)):
        print(speaker_list[speaker_id], '-----------------------------', '\n')
        for f_name in os.listdir(os.path.join(path, speaker_list[speaker_id])):
            if f_name.endswith('.ult') and (('A' in f_name) or ('B' in f_name)):
                ## Read ultrasound files
                ult = read_ultrasound_file(os.path.join(path, speaker_list[speaker_id], f_name))

                ## Read parameters
                param_df = parse_parameter_file(os.path.join(path, speaker_list[speaker_id], f_name[:-4]) + '.param')
                TimeInSecsOfFirstFrame = param_df['TimeInSecsOfFirstFrame'].value
                FramesPerSec = param_df['FramesPerSec'].value

                ## Reshape ult data
                ult_3d = ult.reshape(-1, int(param_df['NumVectors'].value), int(param_df['PixPerVector'].value))

                ## read phone label file
                phonelabel_file = '/media/xys/work/UltraSuite/labels-uxtd-uxssd-upx/uxtd/phone_labels/lab/{}-{}.lab'.format(
                    speaker_list[speaker_id], f_name[:-4])
                pd_phone = pd.read_table(phonelabel_file, sep=' ', index_col=None, header=None,
                                         names=["begin", 'end', 'phonetype'])
                for row in pd_phone.itertuples():
                    phonetype = getattr(row, 'phonetype')
                    label = None
                    if phonetype in Bilabial_Labiodental:
                        label = 0
                    elif phonetype in Dental_Alveolar:
                        label = 1
                    elif phonetype in Velar:
                        label = 2
                    elif phonetype in Approximant:
                        label = 3
                    if label != None:
                        begin = getattr(row, 'begin')
                        end = getattr(row, 'end')
                        beginFrame = max((begin / 1e7 - TimeInSecsOfFirstFrame) * FramesPerSec, 1)
                        endframe = max((end / 1e7 - TimeInSecsOfFirstFrame) * FramesPerSec, 1)
                        centerframe = int((beginFrame + endframe) // 2)

                        im = Image.fromarray(ult_3d[centerframe])
                        im.save(os.path.join(savepath, f'{speaker_list[speaker_id]}', f'{label}', f'{f_name[:-4]}_{row.Index}.jpg'))
    for speaker_id in range(len(speaker_list)):
        dataset_div_folder( os.path.join(savepath, f'{speaker_list[speaker_id]}') )


def make_speakerIndependent():
    path = r"/media/xys/work/UltraSuite/speakerDependentImgs/"
    savepath = r'/media/xys/work/UltraSuite/speakerIndependentImgs/'

    speaker_list = os.listdir(path)
    speaker_list.sort(key=lambda x: int(x[:2]))

    for speaker_id in range(len(speaker_list)):
        for label in range(4):
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'train', f'{label}')
            os.makedirs(savedir, exist_ok=True)
            # savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'valid', f'{label}')
            # os.makedirs(savedir)
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'test', f'{label}')
            os.makedirs(savedir, exist_ok=True)

    for speaker_id_independent in range(len(speaker_list)):
        for speaker_id_dep in range(len(speaker_list)):
            if speaker_id_dep == speaker_id_independent:
                for label in range(4):
                    for filename in os.listdir(  os.path.join(path, speaker_list[speaker_id_dep], f'{label}' ) ):
                        src = os.path.join( path, speaker_list[speaker_id_dep], f'{label}', filename )
                        tgt = os.path.join(savepath, speaker_list[speaker_id_independent], 'test', f'{label}', f'{speaker_list[speaker_id_dep]}_{filename}' )
                        shutil.copyfile(src, tgt)
            else:
                for label in range(4):
                    for filename in os.listdir(  os.path.join(path, speaker_list[speaker_id_dep], f'{label}' ) ):
                        src = os.path.join( path, speaker_list[speaker_id_dep], f'{label}', filename )
                        tgt = os.path.join(savepath, speaker_list[speaker_id_independent], 'train', f'{label}', f'{speaker_list[speaker_id_dep]}_{filename}' )
                        shutil.copyfile(src, tgt)



def make_speakerAdapt():
    path = r"/media/xys/work/UltraSuite/speakerDependentImgs/"
    savepath = r'/media/xys/work/UltraSuite/speakerAdaptImgs/'

    speaker_list = os.listdir(path)
    speaker_list.sort(key=lambda x: int(x[:2]))

    for speaker_id in range(len(speaker_list)):
        for label in range(4):
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'train', f'{label}')
            os.makedirs(savedir, exist_ok=True)
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'finetune', f'{label}')
            os.makedirs(savedir, exist_ok=True)
            savedir = os.path.join(savepath, f'{speaker_list[speaker_id]}', 'test', f'{label}')
            os.makedirs(savedir, exist_ok=True)

    for speaker_id_adapt in range(len(speaker_list)):
        for speaker_id_dep in range(len(speaker_list)):
            if speaker_id_dep == speaker_id_adapt:
                for label in range(4):
                    filename_list = os.listdir(  os.path.join(path, speaker_list[speaker_id_dep], f'{label}' ) )
                    random.shuffle(filename_list)
                    seg = int( len(filename_list) * 0.2)
                    filename_test_list = filename_list[0:seg]
                    filename_finetune_list = filename_list[seg:]
                    for filename in filename_finetune_list:
                        src = os.path.join( path, speaker_list[speaker_id_dep], f'{label}', filename )
                        tgt = os.path.join(savepath, speaker_list[speaker_id_adapt], 'finetune', f'{label}', f'{speaker_list[speaker_id_dep]}_{filename}' )
                        shutil.copyfile(src, tgt)
                    for filename in filename_test_list:
                        src = os.path.join( path, speaker_list[speaker_id_dep], f'{label}', filename )
                        tgt = os.path.join(savepath, speaker_list[speaker_id_adapt], 'test', f'{label}', f'{speaker_list[speaker_id_dep]}_{filename}' )
                        shutil.copyfile(src, tgt)
            else:
                for label in range(4):
                    for filename in os.listdir(  os.path.join(path, speaker_list[speaker_id_dep], f'{label}' ) ):
                        src = os.path.join( path, speaker_list[speaker_id_dep], f'{label}', filename )
                        tgt = os.path.join(savepath, speaker_list[speaker_id_adapt], 'train', f'{label}', f'{speaker_list[speaker_id_dep]}_{filename}' )
                        shutil.copyfile(src, tgt)



if __name__ == '__main__':
    # make_multispeaker()
    # make_speakerDependent()
    # make_speakerIndependent()
    make_speakerAdapt()