import torch
from torchvision.utils import save_image
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(labels, preds, num_classes, return_matrix=False):

    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for ii in range(preds.shape[0]):
        count_matrix[preds[ii], labels[ii]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]

    acc = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum().astype(np.float32) / preds.shape[0]

    if return_matrix:
        return acc, (count_matrix, reassignment)
    else:
        return acc


@torch.no_grad()
def valid(model, data_gen, num_classes, device):
    with torch.no_grad():
        preds = []
        labels = []

        for (image, label) in data_gen:
            image = image.to(device)
            pred = model(image, False).argmax(-1)
            preds.append(pred)
            labels.append(label)

        preds = torch.hstack(preds).cpu().numpy()
        labels = torch.hstack(labels).numpy()

        NMI = metrics.normalized_mutual_info_score(labels, preds) * 100
        AMI = metrics.adjusted_mutual_info_score(labels, preds) * 100
        ARI = metrics.adjusted_rand_score(labels, preds) * 100
        ACC = clustering_accuracy(labels, preds, num_classes) * 100

    return NMI, AMI, ARI, ACC


@torch.no_grad()
def linear_valid(model, data_gen, device):
    with torch.no_grad():

        preds = []
        labels = []

        for (image, label) in data_gen:
            image = image.to(device)
            pred = model(image)
            preds.append(pred)
            labels.append(label)

        preds = torch.vstack(preds).cpu().numpy()
        labels = torch.hstack(labels).numpy()

        Top1 = metrics.top_k_accuracy_score(labels, preds, k=1) * 100
        Top3 = metrics.top_k_accuracy_score(labels, preds, k=3) * 100

    return Top1, Top3


@torch.no_grad()
def rank_valid(model, data_gen, device, num_classes):
    with torch.no_grad():

        preds = []
        labels = []

        for (image, label) in data_gen:
            image = image.to(device)
            pred = model(image)
            preds.append(pred)
            labels.append(label)

        preds = torch.vstack(preds).cpu().numpy()
        labels = torch.hstack(labels).numpy()

        accs = []

        for c in range(num_classes):
            idx = labels == c
            Top1 = metrics.top_k_accuracy_score(labels[idx], preds[idx], k=1) * 100
            accs.append(Top1)
    
    return accs

def get_images(data_gen, c, num_pics):

    pics = []

    for (image, label) in data_gen:
        if label == c:
            pics.append(image)
        if len(pics) >= num_pics:
            break

    return pics

def save_images(images, filename):
    save_image(images, filename)
