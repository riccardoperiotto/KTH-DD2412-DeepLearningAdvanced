# Helper function (Utils)
import torch
import numpy as np
import yaml
from easydict import EasyDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)

    return config


def split_(output, num):
    output1 = []
    output2 = []

    for o in output:
        o1, o2 = torch.chunk(o, num)
        output1.append(o1)
        output2.append(o2)

    return output1, output2


def split(output, num):
    """
    input: (heads, features)
    output: (heads, features split in different transformations)
    """
    split_outputs = [torch.chunk(o, num) for o in output]
    return split_outputs


# modified from DINO
class CosineSchedulerWithWarmup:
    def __init__(self, base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):

        warmup_schedule = np.array([])
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

        cosine_epoch = np.arange(epochs - warmup_epochs)
        final_value = base_value if final_value is None else final_value
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * cosine_epoch / len(cosine_epoch)))
        schedule = np.concatenate((warmup_schedule, schedule)) / start_warmup_value

        assert len(schedule) == epochs
        self.schedule = schedule

    def __call__(self, epoch):
        return self.schedule[epoch]


@torch.no_grad()
def get_visualization_predictions(model, data_gen, num_classes, device):
    with torch.no_grad():
        preds = []
        labels = []
        visualize_images = [[] for i in range(num_classes)]

        for (image, label) in data_gen:
            image, visualize_image = image[0].to(device), image[1]

            pred = model(image).argmax(-1)
            preds.append(pred)
            labels.append(label)

            for l, vi_image in zip(label, visualize_image):
                visualize_images[l].append(vi_image.unsqueeze(0))

        preds = torch.hstack(preds).cpu().numpy()
        labels = torch.hstack(labels).numpy()

        return preds, labels, visualize_images

