import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os


def getSVHN(batch_size, TF, data_root='data', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='train', download=True, transform=TF), batch_size=batch_size, shuffle=True)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=data_root, split='test', download=True, transform=TF,), batch_size=batch_size, shuffle=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR10(batch_size, TF, data_root='data', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=True, download=True, transform=TF), batch_size=batch_size, shuffle=True)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_root, train=False, download=True, transform=TF), batch_size=batch_size, shuffle=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, TF, data_root='data', TTF=None, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF, target_transform=TTF), batch_size=batch_size, shuffle=True)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF, target_transform=TTF), batch_size=batch_size, shuffle=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getIMAGENET1K(batch_size, TF, data_root='data', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'imagenet1k'))
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(data_root, 'train'), transform=TF),
            batch_size=batch_size, shuffle=True)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(data_root, 'val'), transform=TF),
            batch_size=batch_size, shuffle=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(args, data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet1k':
        train_loader, test_loader = getIMAGENET1K(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=8)
    return train_loader, test_loader


def getNonTargetDataSet(args, data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=lambda x: 0, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == 'cifar10_64':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=transforms.Compose([transforms.Resize((64, 64)), input_TF]),
        data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100_64':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=transforms.Compose([transforms.Resize((64, 64)), input_TF]),
        data_root=dataroot, TTF=lambda x: 0, num_workers=1)
    elif data_type == 'svhn_64':
        _, test_loader = getSVHN(batch_size=batch_size, TF=transforms.Compose([transforms.Resize((64, 64)), input_TF]),
        data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet-o-64':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'imagenet-o-64'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False)
    elif data_type == 'imagenet-o':
        dataroot = os.path.expanduser(os.path.join('data', 'imagenet-o'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=8)
    return test_loader
