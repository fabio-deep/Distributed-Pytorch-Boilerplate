import os, torch, torchvision
import numpy as np
import scipy.io as sio
from PIL import Image

from utils import *
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

def cifar10(args, dataset_paths):

    transf = {'train': transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
            transforms.ToTensor(),
            # Standardize()]),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                (0.24703223, 0.24348513, 0.26158784))]),
        'test':  transforms.Compose([
            transforms.ToTensor(),
            # Standardize()])}
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                (0.24703223, 0.24348513, 0.26158784))])}

    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.CIFAR10(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_sample_weights = SampleWeights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10).reshape(-1))

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
            (0.24703223, 0.24348513, 0.26158784))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        #Standardize()])
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
            (0.24703223, 0.24348513, 0.26158784))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    # weighted sampler weights for new training set
    sample_weights = SampleWeights(datasets['train'].targets)

    config = {'train': WeightedRandomSampler(sample_weights,
        num_samples=len(sample_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_sample_weights,
        num_samples=len(f_sample_weights), replacement=True),
        'valid': None, 'test': None}

    if args.distributed:
        config = {'train': DistributedSampler(datasets['train']),
            'train_valid': DistributedSampler(datasets['train_valid']),
            'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
        num_workers=8, pin_memory=True, drop_last=True,
        batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

def svhn(args, dataset_paths):

    transf = {'train': transforms.Compose([
            # transforms.RandomApply([
            #     transforms.RandomAffine(30, shear=True)], p=0.5),
            transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
            transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
            transforms.ToTensor(),
            # Standardize()]),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                (0.19803012, 0.20101562, 0.19703614))]),
        # 'extra': transforms.Compose([
        #     # transforms.RandomApply([
        #     #     transforms.RandomAffine(30, shear=True)], p=0.5),
        #     transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        #     transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
        #     transforms.ToTensor(),
        #     # Standardize()]),
        #     transforms.Normalize((0.4379, 0.4441, 0.4734), (0.1202, 0.1232, 0.1054))]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # Standardize()])}
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                (0.19803012, 0.20101562, 0.19703614))])}

    # config = {'train': True, 'extra': True, 'test': False}
    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.SVHN(root=dataset_paths[i], transform=transf[i],
        split=i, download=True) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_sample_weights = SampleWeights(datasets['train'].labels)

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].labels,
        n_classes=10,
        n_samples_per_class=np.unique(
            datasets['test'].labels, return_counts=True)[1] // 3) # fraction of test set per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomApply([
        #     transforms.RandomAffine(30, shear=True)], p=0.5),
        transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
            (0.19803012, 0.20101562, 0.19703614))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
            (0.19803012, 0.20101562, 0.19703614))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make channels last and convert to np arrays
    data['train'] = np.moveaxis(np.array(data['train']), 1, -1)
    data['valid'] = np.moveaxis(np.array(data['valid']), 1, -1)

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    # weighted sampler weights for new training set
    sample_weights = SampleWeights(datasets['train'].labels)

    config = {'train': WeightedRandomSampler(sample_weights,
        num_samples=len(sample_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_sample_weights,
        num_samples=len(f_sample_weights), replacement=True),
        'valid': None, 'test': None}

    if args.distributed:
        config = {'train': DistributedSampler(datasets['train']),
            'train_valid': DistributedSampler(datasets['train_valid']),
            'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
        num_workers=8, pin_memory=True, drop_last=True,
        batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

def fashionmnist(args, dataset_paths):

    transf = {'train': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                # Standardize()]),
                transforms.Normalize((0.28604059,),(0.35302424,))]),
        'test':  transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                # transforms.CenterCrop((args.crop_dim, args.crop_dim)),
                transforms.ToTensor(),
                # Standardize()])}
                transforms.Normalize((0.28604059,),(0.35302424,))])}

    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.FashionMNIST(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_sample_weights = SampleWeights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.unique(
            datasets['test'].targets, return_counts=True)[1] // 2) # half of test set per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.28604059,),(0.35302424,))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
        transforms.ToTensor(),
        # Standardize()])
        transforms.Normalize((0.28604059,),(0.35302424,))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    # weighted sampler weights for new training set
    sample_weights = SampleWeights(datasets['train'].labels)

    config = {'train': WeightedRandomSampler(sample_weights,
        num_samples=len(sample_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_sample_weights,
        num_samples=len(f_sample_weights), replacement=True),
        'valid': None, 'test': None}

    if args.distributed:
        config = {'train': DistributedSampler(datasets['train']),
            'train_valid': DistributedSampler(datasets['train_valid']),
            'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
        num_workers=8, pin_memory=True, drop_last=True,
        batch_size=args.batch_size) for i in config.keys()}

    return dataloaders

def mnist(args, dataset_paths):

    transf = {'train': transforms.Compose([
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))
                ]),
        'test':  transforms.Compose([
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))])}

    config = {'train': True, 'test': False}
    datasets = {i: torchvision.datasets.MNIST(root=dataset_paths[i], transform=transf[i],
        train=config[i], download=True) for i in config.keys()}

    # split train into train and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].targets,
        n_classes=10,
        n_samples_per_class=np.repeat(500, 10)) # 500 per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((args.crop_dim, args.crop_dim), padding=args.padding),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(np.maximum(0, (args.crop_dim-28) // 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.13066047,), (0.30810780,))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    if args.distributed:
        config = {'train': DistributedSampler(datasets['train']),
            'train_valid': DistributedSampler(datasets['train_valid']),
            'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], sampler=config[i],
        num_workers=8, pin_memory=True, drop_last=True,
        batch_size=args.batch_size) for i in config.keys()}

    return dataloaders
