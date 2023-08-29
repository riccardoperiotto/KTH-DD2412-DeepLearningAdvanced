import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import random_split


def get_crop_size(dset_name):
    if dset_name == 'mnist':
        crop_size = 28
        local_crop_size = 14
    elif dset_name == 'cifar10' or dset_name == 'cifar100' or dset_name == 'cifar20':
        crop_size = 32
        local_crop_size = 32
    elif dset_name == 'stl10':
        crop_size = 96
        local_crop_size = 42
    elif dset_name == 'tiny_imagenet':
        crop_size = 224
        local_crop_size = 96

    return crop_size, local_crop_size


class Augmentation:
    def __init__(self, train=True, dset_name='cifar10', multi_crop=0):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.normalize = normalize

        # flip, colorjitter, grayscale in BYOL
        base_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2)])

        crop_size, local_crop_size = get_crop_size(dset_name)
        global_scale = (0.5, 1.0) if multi_crop else (0.2, 1.0)
        local_scale = (0.2, 0.5)

        # additional transformation for mnist
        if dset_name == 'mnist':
            gray_to_color = T.Grayscale(3)
            base_transforms = T.Compose([gray_to_color, base_transforms])
            self.normalize = T.Compose([gray_to_color, normalize])

        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(size=(crop_size, crop_size), scale=global_scale, interpolation=Image.BICUBIC),
            base_transforms,
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.1),
            normalize])

        self.global_transform2 = T.Compose([
              T.RandomResizedCrop(size=(crop_size, crop_size), scale=global_scale, interpolation=Image.BICUBIC),
              base_transforms,
              T.RandomApply([T.GaussianBlur(kernel_size=5)], p=1.0),
              T.RandomSolarize(threshold=128, p=0.2),
              normalize])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(size=(local_crop_size, local_crop_size), scale=local_scale, interpolation=Image.BICUBIC),
            base_transforms,
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
            normalize])

        self.train = train
        self.multi_crop = multi_crop

    def __call__(self, images):

        # image go through same base transform
        # then creat different views

        outputs = []
        if self.train:
            outputs.append(self.global_transform1(images))
            outputs.append(self.global_transform2(images))

            for _ in range(self.multi_crop):
                outputs.append(self.local_transform(images))

            return outputs

        else:
            return self.normalize(images)


class LinearAugmentation:
    def __init__(self, train=True, dset_name='cifar10'):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.normalize = normalize

        crop_size = get_crop_size(dset_name)

        # additional transformation for mnist
        if dset_name == 'mnist':
            gray_to_color = T.Grayscale(3)
            self.normalize = T.Compose([gray_to_color, normalize])

        self.lin_train = T.Compose([
            T.RandomResizedCrop(crop_size),
            T.RandomHorizontalFlip(),
            self.normalize
        ])

        self.lin_val = T.Compose([
            self.normalize
        ])

        self.train = train

    def __call__(self, images):

        if self.train:
            return self.lin_train(images)
        else:
            return self.lin_val(images)


class VisualizeAugmentation:
    def __init__(self, dset_name='cifar10'):

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.normalize = normalize

        # additional transformation for mnist
        if dset_name == 'mnist':
            gray_to_color = T.Grayscale(3)
            self.normalize = T.Compose([gray_to_color, normalize])

        self.vi_transform = T.ToTensor()

    def __call__(self, images):

        return [self.normalize(images), self.vi_transform(images)]


class Cifar20TargetTransform:
    def __init__(self):
        self.dict = \
            {0: 4,
             1: 1,
             2: 14,
             3: 8,
             4: 0,
             5: 6,
             6: 7,
             7: 7,
             8: 18,
             9: 3,
             10: 3,
             11: 14,
             12: 9,
             13: 18,
             14: 7,
             15: 11,
             16: 3,
             17: 9,
             18: 7,
             19: 11,
             20: 6,
             21: 11,
             22: 5,
             23: 10,
             24: 7,
             25: 6,
             26: 13,
             27: 15,
             28: 3,
             29: 15,
             30: 0,
             31: 11,
             32: 1,
             33: 10,
             34: 12,
             35: 14,
             36: 16,
             37: 9,
             38: 11,
             39: 5,
             40: 5,
             41: 19,
             42: 8,
             43: 8,
             44: 15,
             45: 13,
             46: 14,
             47: 17,
             48: 18,
             49: 10,
             50: 16,
             51: 4,
             52: 17,
             53: 4,
             54: 2,
             55: 0,
             56: 17,
             57: 4,
             58: 18,
             59: 17,
             60: 10,
             61: 3,
             62: 2,
             63: 12,
             64: 12,
             65: 16,
             66: 12,
             67: 1,
             68: 9,
             69: 19,
             70: 2,
             71: 10,
             72: 0,
             73: 1,
             74: 16,
             75: 12,
             76: 9,
             77: 13,
             78: 15,
             79: 13,
             80: 16,
             81: 19,
             82: 2,
             83: 4,
             84: 6,
             85: 19,
             86: 5,
             87: 5,
             88: 8,
             89: 19,
             90: 18,
             91: 1,
             92: 2,
             93: 15,
             94: 6,
             95: 0,
             96: 17,
             97: 8,
             98: 14,
             99: 13}

    def __call__(self, target):
        new_target = self.dict[target]
        return new_target


def get_data(dset_name, batch_size, multi_crop, mode='train'):

    if mode == 'train':
        aug_train = Augmentation(train=True, dset_name=dset_name, multi_crop=multi_crop)
        aug_test = Augmentation(train=False, dset_name=dset_name, multi_crop=0)
    elif mode == 'linear_probing':
        aug_train = LinearAugmentation(train=True, dset_name=dset_name)
        aug_test = LinearAugmentation(train=False, dset_name=dset_name)
    elif mode == 'visualize':
        aug_train = VisualizeAugmentation(dset_name=dset_name)
        aug_test = VisualizeAugmentation(dset_name=dset_name)

    # data download
    if dset_name == 'mnist':
        train_data = dsets.MNIST(root='./data', train=True, transform=aug_train, download=True)
        test_data = dsets.MNIST(root='./data', train=False, transform=aug_test)
    elif dset_name == 'cifar10':
        train_data = dsets.CIFAR10(root='./data', train=True, transform=aug_train, download=True)
        test_data = dsets.CIFAR10(root='./data', train=False, transform=aug_test)
    elif dset_name == 'cifar100':
        train_data = dsets.CIFAR100(root='./data', train=True, transform=aug_train, download=True)
        test_data = dsets.CIFAR100(root='./data', train=False, transform=aug_test)
    elif dset_name == 'cifar20':
        train_data = dsets.CIFAR100(root='./data', train=True, transform=aug_train,\
                                    target_transform=Cifar20TargetTransform(), download=True)
        test_data = dsets.CIFAR100(root='./data', train=False, transform=aug_test, target_transform=Cifar20TargetTransform())
    elif dset_name == 'stl10':
        train_data = dsets.STL10(root='./data', split="train", transform=aug_train, download=True)
        test_data = dsets.STL10(root='./data', split="test", transform=aug_test)
    elif dset_name == 'tiny_imagenet':
        train_data, _ = random_split(dsets.ImageFolder('data/tiny-imagenet-200/train', transform=aug_train), [90000, 10000])
        _, test_data = random_split(dsets.ImageFolder('data/tiny-imagenet-200/train', transform=aug_test), [90000, 10000])

    # data load
    train_gen = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_gen = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_gen, test_gen
