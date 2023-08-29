
import torch
from valid import rank_valid, get_images, save_images
from data import get_data
from utils import get_config
from linear_probing import load_model

import os
import random

import argparse



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
    _, test_gen = get_data(args.dset, args.batch_size, args.multi_crop, mode='linear_probing')

    file_names = os.listdir(pre_args.folder)
    file_names = sorted(file_names, key=lambda x: int(x[-7:-4]))
    for f in file_names:
        file_path = os.path.join(pre_args.folder, f)
        model = load_model(file_path, device, args)
        print(f)

        accs = rank_valid(model, test_gen, device, args.num_classes)
        for (c, acc) in enumerate(accs):
            print("classs "+str(c)+"   acc: " + str(acc))
            fname = "pics_C" + str(c) + "_acc:" + str(acc) + "_.png"
            pics = get_images(test_gen, c, 16)
            save_images(pics, fname)