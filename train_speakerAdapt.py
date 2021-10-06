import argparse
import os
import math
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import logging
import time
import numpy as np

from models import SimpleCNN
from data import dataloader
from losses import FocalLoss, GHMC



def get_logger(filename='logger-speakerAdpt',level='info', outPath='./'):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s - %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    format_str = logging.Formatter(fmt,datefmt)

    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    timeStr = time.strftime("%Y-%m-%d--%H-%M-%S",time.localtime())
    th = logging.FileHandler(filename=os.path.join(outPath,filename+'--'+timeStr), encoding='utf-8')
    th.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger

def get_args():
    selfsup_file = 'selfsup-SimSiam-MyModel.pt'
    # selfsup_file = ''
    lr_base = 1e-4
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr_classifier', default=lr_base, type=float, help='learning rate for classifier')
    parser.add_argument('--lr_backbone', default=lr_base, type=float, help='learning rate for backbone')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--selfsup_file', default=selfsup_file, type=str, help='pretrained checkpoint file')
    parser.add_argument('--model', default="MyModel", type=str,
                        help='model type (default: MyModel)')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='channels of input images, 1: gray image; 3: pseudo colorful image')
    parser.add_argument('--imagenetpretrained', default=False, type=bool,
                        help='whether to load the imagenet pretrained params')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--warmup_epoch', default=3, type=int, help='warmup epochs')
    parser.add_argument('--max_epoch', default=10, type=int, help='total epochs to run')
    parser.add_argument('--freeze_epochs', default=0, type=int, help='freeze the backbone for some epochs, then unfreeze it')

    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=2020, type=int, help='random seed')

    parser.add_argument('--augment', default=True,
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=0.5e-3, type=float, help='weight decay') #0.5e-3

    parser.add_argument('--losstype', default='CE_focalloss', type=str, help=' CE_focalloss, CE')
    parser.add_argument('--loss_lambda', default=2.0, type=float, help='used when CE_focalloss or CE_GHMC')

    args = parser.parse_args()

    logger.info(dict(args._get_kwargs()))
    return args


def checkpoint(args, net, acc, epoch):
    # Save checkpoint.WO TIAN
    # print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def cosinLR(optimizer, warmup_t, T):
    t = warmup_t
    n_t = 0.5
    min_ratio = 0.01
    lambda1 = lambda epoch: ((1-min_ratio) * epoch / t + min_ratio) if epoch < t \
        else min_ratio if n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t))) < min_ratio \
        else n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t)))
    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    return sched

def train(epoch, lr_scheduler, net, trainloader, criterion, optimizer, use_cuda, args):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    # reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

###############################################################
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if args.losstype.startswith('CE_'):
            criterion_ce = nn.CrossEntropyLoss()
            loss_ce = criterion_ce(outputs, targets)
            loss = loss * args.loss_lambda + loss_ce
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        #########################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()

    return (train_loss/(batch_idx+1), 100.*correct/total)


def modelfinetune(epoch, lr_scheduler, net, trainloader, criterion, optimizer, use_cuda, args):
    net.train()
    train_loss = 0
    # reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

###############################################################
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if args.losstype.startswith('CE_'):
            criterion_ce = nn.CrossEntropyLoss()
            loss_ce = criterion_ce(outputs, targets)
            loss = loss + loss_ce
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        #########################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()

    return (train_loss/(batch_idx+1), 100.*correct/total)

def modeltest(epoch, args, net, criterion, testloader, use_cuda, best_acc):
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


    acc = 100.*correct/total
    '''if epoch == args.max_epoch - 1 or acc > best_acc:
        checkpoint(args, net, acc, epoch)'''
    if acc > best_acc:
        best_acc = acc
    return (test_loss/(batch_idx+1), 100.*correct/total, best_acc)

def worker_oneSpk(args, dataFolder):
    use_cuda = torch.cuda.is_available()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0

    folder_train = os.path.join(dataFolder, 'train')
    loader_train = dataloader.build_dataloader_folder(folder_train, args.batch_size, 'train', channels=args.input_channels, num_workers=2)

    folder_finetune = os.path.join(dataFolder, 'finetune')
    loader_finetune = dataloader.build_dataloader_folder(folder_finetune, args.batch_size, 'train', channels=args.input_channels, num_workers=2)

    folder_test  = os.path.join(dataFolder, 'test')
    loader_test  = dataloader.build_dataloader_folder(folder_test,  args.batch_size, 'test',  channels=args.input_channels, num_workers=2)
    test_exampleNum = len(loader_test.dataset)

    if args.model == 'MyModel_no1Dbn':
        net = SimpleCNN.MyModel_no1Dbn()
    elif args.model == 'MyModel_no2Dbn':
        net = SimpleCNN.MyModel_no2Dbn()
    else:
        net = SimpleCNN.MyModel()

    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        # logger.info('Using CUDA..')
    if args.selfsup_file != '':
        net.load_state_dict( torch.load(args.selfsup_file) )

    # criterion = nn.CrossEntropyLoss()
    if args.losstype == 'CE_focalloss':
        criterion = FocalLoss(gamma=0.5)
    else:
        criterion = nn.CrossEntropyLoss()

    params_backbone   = [param for name, param in net.named_parameters() if 'fc' not in name]
    params_classifier = [param for name, param in net.named_parameters() if 'fc' in name]
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)  # lr=0.000078
    optimizer = optim.SGD(  [{'params':params_backbone, 'lr': args.lr_backbone},
                            {'params': params_classifier, 'lr': args.lr_classifier} ],
                            momentum=0.9, weight_decay=args.decay)  # lr=0.000078
    lr_scheduler = cosinLR(optimizer, args.warmup_epoch, args.max_epoch)

    # freeze the backbone begin
    for name, param in net.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    for epoch in range(start_epoch, args.max_epoch):
        if epoch == args.freeze_epochs: # unfreeze the backbone
            for name, param in net.named_parameters():
                param.requires_grad = True

        train_loss, train_acc = train(epoch, lr_scheduler, net, loader_train, criterion, optimizer, use_cuda, args)
        test_loss, test_acc, best_acc = modeltest(epoch, args, net, criterion, loader_test, use_cuda, best_acc)
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_classifier = optimizer.param_groups[1]['lr']
        logger.info('train    {}/{}:  lr_backbone {:.8f}, lr_classifier {:.8f},  train_loss {:.4f}, train_acc {:.4f}, test_loss {:.4f}, test_acc {:.4f}, best_acc {:.4f}'.
                    format(epoch, args.max_epoch, lr_backbone, lr_classifier, train_loss, train_acc, test_loss, test_acc, best_acc))

    optimizer_ft = optim.SGD([{'params': params_backbone, 'lr': args.lr_backbone},
                           {'params': params_classifier, 'lr': args.lr_classifier}],
                          momentum=0.9, weight_decay=args.decay)  # lr=0.000078
    lr_scheduler_ft = cosinLR(optimizer_ft, args.warmup_epoch, args.max_epoch)
    for epoch in range(start_epoch, args.max_epoch):
        finetune_loss, finetune_acc = modelfinetune(epoch, lr_scheduler_ft, net, loader_finetune, criterion, optimizer_ft, use_cuda, args)
        test_loss, test_acc, best_acc = modeltest(epoch, args, net, criterion, loader_test, use_cuda, best_acc)
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_classifier = optimizer.param_groups[1]['lr']
        logger.info(
            'finetune {}/{}:  lr_backbone {:.8f}, lr_classifier {:.8f},  finetune_loss {:.4f}, finetune_acc {:.4f}, test_loss {:.4f}, test_acc {:.4f}, best_acc {:.4f}'.
            format(epoch, args.max_epoch, lr_backbone, lr_classifier, finetune_loss, finetune_acc, test_loss, test_acc, best_acc))

    return test_acc, best_acc, test_exampleNum


def main():
    args = get_args()
    if args.seed != 0:
        torch.manual_seed(args.seed)

    test_acc_arr = []
    best_acc_arr = []
    test_exampleNum_arr = []

    path = r"/media/xys/work/UltraSuite/speakerAdaptImgs/"
    speaker_list = os.listdir(path)
    speaker_list.sort(key=lambda x:int(x[:2]))
    for speaker_id in range(len(speaker_list)):
        logger.info(f'Working for speaker {speaker_list[speaker_id]} .........')
        dataFolder_spk = os.path.join(path, f'{speaker_list[speaker_id]}' )
        test_acc, best_acc, test_exampleNum = worker_oneSpk(args, dataFolder_spk)
        test_acc_arr.append(test_acc)
        best_acc_arr.append(best_acc)
        test_exampleNum_arr.append(test_exampleNum)
    test_acc_mean = np.mean( np.array(test_acc_arr) )
    best_acc_mean = np.mean( np.array(best_acc_arr) )

    logger.info(f'test_acc_arr:  {test_acc_arr}')
    logger.info(f'best_acc_arr:  {best_acc_arr}')
    logger.info(f'test_acc_mean: {test_acc_mean};   best_acc_mean: {best_acc_mean}')
    test_acc_num = np.array(test_acc_arr) * np.array(test_exampleNum_arr)
    test_acc_num = np.sum(test_acc_num)
    test_acc_OA  = test_acc_num / np.sum(np.array(test_exampleNum_arr))
    logger.info(f'test_acc_OA: {test_acc_OA}')




if __name__ == '__main__':
    savepathBase = './output'
    if not os.path.isdir(savepathBase):
        os.makedirs(savepathBase)
    logger = get_logger(outPath=savepathBase)
    for i in range(1):
        main()