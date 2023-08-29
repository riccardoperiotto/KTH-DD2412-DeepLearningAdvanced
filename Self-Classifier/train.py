import torch
import torch.optim as optim
import torch.nn.functional as F

from torchlars import LARS
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from valid import valid
from data import get_data
from utils import split, split_, CosineSchedulerWithWarmup, get_config, AverageMeter
from model import Model

import os
import random
from torch.utils.tensorboard import SummaryWriter
import argparse


class SelfClassifierLoss:

    def __init__(self, t_row, t_col):
        self.t_row = t_row
        self.t_col = t_col

    def __call__(self, out1, out2):
        L = 0

        for s1, s2 in zip(out1, out2):
            N, C = s1.shape
            log_y_x1 = torch.log(N / C * F.normalize(F.softmax(s1 / self.t_row, dim=1), p=1, dim=0))
            log_y_x2 = torch.log(N / C * F.normalize(F.softmax(s2 / self.t_row, dim=1), p=1, dim=0))
            y_x1 = F.normalize(F.softmax(s1 / self.t_col, dim=0), p=1, dim=1)
            y_x2 = F.normalize(F.softmax(s2 / self.t_col, dim=0), p=1, dim=1)
            l1 = -torch.mean(torch.sum(y_x2 * log_y_x1, dim=1))
            l2 = -torch.mean(torch.sum(y_x1 * log_y_x2, dim=1))
            L += (l1 + l2) / 2
        return L


class SelfClassifierLossWithMulticrop:

    def __init__(self, t_row, t_col):
        self.t_row = t_row
        self.t_col = t_col

    def __call__(self, outputs):

        L = 0
        loss_terms = 0

        for head_outputs in outputs:  # iterate over each classifier heads

            target = []
            N, C = head_outputs[0].shape

            for idx_i, output in enumerate(head_outputs):  # iterate over different transformations
                y_x = F.normalize(F.softmax(output / self.t_col, dim=0), p=1, dim=1)
                target.append(y_x)

            for idx_j, output in enumerate(head_outputs):
                log_y_x = torch.log(N / C * F.normalize(F.softmax(output / self.t_row, dim=1), p=1, dim=0))

                for idx_i, y_x in enumerate(target):

                    if idx_i == idx_j or (idx_i > 1 and idx_j > 1):
                        continue
                    loss_ij = - torch.mean(torch.sum(y_x * log_y_x, dim=1))
                    L += loss_ij
                    loss_terms += 1

        L /= loss_terms
        return L


class NaiveCrossEntropy():
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, out1, out2):
        L = 0
        for s1, s2 in zip(out1, out2):
            L += 0.5 * (self.loss(s1, s2.softmax(dim=1)) + self.loss(s2, s1.softmax(dim=1)))
        return L


def train(model, train_gen, test_gen, device, args):

    writer = SummaryWriter()

    if args.optimizer == 'sgd':
        base_optimizer = optim.SGD(model.parameters(), lr=args.start_warmup_value,
                                   momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        base_optimizer = optim.AdamW(model.parameters(), lr=args.start_warmup_value,
                                     weight_decay=args.weight_decay)
    if args.lars:
        optimizer = LARS(optimizer=base_optimizer, trust_coef=0.001)
    else:
        optimizer = base_optimizer

    scheduler_function = CosineSchedulerWithWarmup(args.lr, args.final_lr, args.num_epochs,
                                                   args.warmup_epochs, args.start_warmup_value)
    scheduler = LambdaLR(optimizer, scheduler_function)

    if not args.multi_crop:
        loss_function = SelfClassifierLoss(args.t_row, args.t_col)
        # loss_function = NaiveCrossEntropy()
    else:
        loss_function = SelfClassifierLossWithMulticrop(args.t_row, args.t_col)

    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(train_gen):
            # train
            model.train

            loss_meter = AverageMeter('Loss', ':.4e')

            if args.multi_crop:
                global_images = Variable(torch.cat(images[:2])).to(device)
                local_images = Variable(torch.cat(images[2:])).to(device)

                optimizer.zero_grad()
                global_outputs = model(global_images)
                local_outputs = model(local_images)
                global_outputs = split(global_outputs, 2)
                local_outputs = split(local_outputs, args.multi_crop)

                outputs = global_outputs + local_outputs

                loss = loss_function(outputs)
                loss.backward()
                optimizer.step()

            else:
                images = Variable(torch.cat(images)).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                outputs1, outputs2 = split_(outputs, 2)
                loss = loss_function(outputs1, outputs2)

                loss_meter.update(loss.item(), images.shape[0]/2)
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
            NMI, AMI, ARI, ACC = valid(model, test_gen, args.num_classes, device)

            writer.add_scalar('NMI/test', NMI, epoch)
            writer.add_scalar('AMI/test', AMI, epoch)
            writer.add_scalar('ARI/test', ARI, epoch)
            writer.add_scalar('ACC/test', ACC, epoch)

            print('testing...')
            print('Epoch [%d/%d], NMI: %.4f, AMI: %.4f, ARI: %.4f, ACC: %.4f' %
                  (epoch + 1, args.num_epochs, NMI, AMI, ARI, ACC))

        # save model
        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists('./save_%s'%args.dset):
                os.makedirs('./save_%s'%args.dset)
            torch.save(model.state_dict(), './save_%s/%s_%s.pth'%(args.dset, args.backbone, epoch+1))

        if (epoch + 1) == args.early_stop:
            torch.save(model.state_dict(), './ablation/batch256.pth')
            return


parser = argparse.ArgumentParser(description='PyTorch self-classifier')
parser.add_argument('-c', '--config_file', metavar='CONFIG', help='which config file to choose')

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

    train_gen, test_gen = get_data(args.dset, args.batch_size, args.multi_crop, mode='train')
    model = Model(args.hidden_dim, args.head_sizes, args.backbone).to(device)

    train(model, train_gen, test_gen, device, args)
