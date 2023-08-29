import torch

from valid import clustering_accuracy
from data import get_data
from utils import get_config, get_visualization_predictions
from model import Model
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

import os
import random
import argparse
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('Agg')
cmap = plt.get_cmap('nipy_spectral')

class VisualizeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = []
        self.model.backbone.register_forward_hook(self.save_outputs_hook())

    def save_outputs_hook(self):
        def fn(_, __, output):
            self.features.append(output)
        return fn

    def get_features(self):
        features = torch.vstack(self.features).squeeze().cpu().numpy()
        self.features = []
        return features

    def forward(self, x):
        x = self.model(x, False)
        return x


def load_model(file_path, device, args):
    model = Model(args.hidden_dim, args.head_sizes, args.backbone).to(device)
    model.load_state_dict(torch.load(file_path))
    model = VisualizeModel(model)

    return model

def get_classes(dset):
    if dset == 'stl10':
        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    elif dset == 'mnist':
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dset == 'cifar10':
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dset == 'cifar20':
        classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', \
                   'household electrical devices', 'househould furniture', 'insects', 'large carnivores', \
                   'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', \
                   'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', \
                   'trees', 'vehicles 1', 'vehicles 2']
    return classes


def visualize(model, test_gen, device, args):
    preds, labels, visualize_images = get_visualization_predictions(model, test_gen, args.num_classes, device)
    features = model.get_features()

    tsne_feats = TSNE(n_components=2, random_state=0).fit_transform(features)
    classes = get_classes(args.dset)

    scatter = plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=labels, cmap=cmap, s=5)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.savefig('./imgs/%s_tsne.png' % args.dset)
    plt.clf()

    acc, (count_matrix, reassignment) = clustering_accuracy(labels, preds, args.num_classes, return_matrix=True)
    acc_per_class = count_matrix[reassignment[:, 0], reassignment[:, 1]] / count_matrix.sum(1)

    num_per_predicted_class = count_matrix.sum(1)

    if not os.path.exists('./imgs'):
        os.makedirs('./imgs')

    plt.bar(reassignment[:, 0], num_per_predicted_class)
    plt.savefig('./imgs/%s.png'%args.dset)

    sorted_idx_i = np.argsort(acc_per_class)[::-1]

    # top 2 and worst 2 classes
    highest_class_idx, lowest_class_idx = sorted_idx_i[:2], sorted_idx_i[-2:]

    for i, idx in enumerate(highest_class_idx):

        visualize_image = np.vstack(visualize_images[idx])
        visualize_image_idx = np.random.randint(0, len(visualize_image), size=9)
        visualize_image = visualize_image[visualize_image_idx]

        grid = make_grid(torch.tensor(visualize_image), nrow=3)
        img = ToPILImage()(grid)

        img.save('./imgs/%s_highest%s.png'%(args.dset, i))

    for i, idx in enumerate(lowest_class_idx):
        visualize_image = np.vstack(visualize_images[idx])
        visualize_image_idx = np.random.randint(0, len(visualize_image), size=9)
        visualize_image = visualize_image[visualize_image_idx]

        grid = make_grid(torch.tensor(visualize_image), nrow=3)
        img = ToPILImage()(grid)

        img.save('./imgs/%s_lowest%s.png' % (args.dset, i))


parser = argparse.ArgumentParser(description='PyTorch self-classifier')
parser.add_argument('-c', '--config_file', metavar='CONFIG', help='which config file to choose')
parser.add_argument('-f', '--file', metavar='FILE', help='which saved model to visualize')


if __name__ == '__main__':

    pre_args = parser.parse_args()
    args = get_config(pre_args.config_file)

    print(pre_args)
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_gen, test_gen = get_data(args.dset, args.batch_size, args.multi_crop, mode='visualize')

    model = load_model(pre_args.file, device, args)
    visualize(model, test_gen, device, args)
