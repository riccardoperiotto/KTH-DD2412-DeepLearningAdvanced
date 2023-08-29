import torch
import torch.optim as optim
import torch.nn as nn

from torchlars import LARS
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from valid import linear_valid
from data import get_data
from utils import CosineSchedulerWithWarmup, get_config, AverageMeter
from model import Model, LCModel

import os
import random
from torch.utils.tensorboard import SummaryWriter
import argparse


def load_model(file_path, device, args):
    model = Model(args.hidden_dim, args.head_sizes, args.backbone).to(device)
    model.load_state_dict(torch.load(file_path))
    model = LCModel(model, args.num_classes).to(device)

    return model


def train(model, train_gen, test_gen, device, args):
    writer = SummaryWriter()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if args.optimizer == 'sgd':
        base_optimizer = optim.SGD(params, lr=args.lr,
                                   momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        base_optimizer = optim.AdamW(params, lr=args.lr,
                                     weight_decay=args.weight_decay)
    if args.lars:
        optimizer = LARS(optimizer=base_optimizer, trust_coef=0.001)
    else:
        optimizer = base_optimizer

    scheduler_function = CosineSchedulerWithWarmup(args.lr, args.final_lr, args.num_epochs,
                                                   args.warmup_epochs, args.start_warmup_value)

    scheduler = LambdaLR(optimizer, scheduler_function)
    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in range(100):
        for i, (images, labels) in enumerate(train_gen):
            # train
            model.train

            loss_meter = AverageMeter('Linear_Loss', ':.4e')
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss_meter.update(loss.item(), images.shape[0])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.train_interval == 0:
            print('Epoch [%d/%d], Loss: %.4f'% (epoch + 1, args.num_epochs, loss_meter.val))
            writer.add_scalar('loss/train', loss_meter.avg, epoch)

        try:
            scheduler.step()
        except:
            pass

        # test
        if (epoch + 1) % args.test_interval == 0:
            model.eval
            Top1, Top3 = linear_valid(model, test_gen, device)

            writer.add_scalar('Top1/test', Top1, epoch)
            writer.add_scalar('Top3/test', Top3, epoch)
            print('Epoch [%d/%d], Top1: %.4f, Top3: %.4f' % (epoch + 1, args.num_epochs, Top1, Top3))


parser = argparse.ArgumentParser(description='PyTorch self-classifier')
parser.add_argument('-c', '--config_file', metavar='CONFIG', help='which config file to choose')
parser.add_argument('-f', '--folder', metavar='FOLDER', help='folder with saved models')

if __name__ == '__main__':

    pre_args = parser.parse_args()
    args = get_config(pre_args.config_file)
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_gen, test_gen = get_data(args.dset, args.batch_size, args.multi_crop, mode='linear_probing')

    file_names = os.listdir(pre_args.folder)
    file_names = sorted(file_names, key=lambda x: int(x[-7:-4]))
    for f in file_names:
        file_path = os.path.join(pre_args.folder, f)
        model = load_model(file_path, device, args)
        print(f)
        train(model, train_gen, test_gen, device, args)
