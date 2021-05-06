import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

def cutout(mask_size, p, cutout_inside=False, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

def getDataloaders(data, config_of_data, splits=['train', 'val', 'test'],
                   aug=True, use_validset=True, data_root='data', batch_size=64, normalized=True,
                   num_workers=3, **kwargs):
    train_loader, val_loader, test_loader = None, None, None

    if data.find('cifar10') >= 0:
        print('loading ' + data)
        print(config_of_data)
        if data.find('cifar100') >= 0:
            d_func = dset.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
        else:
            d_func = dset.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                cutout(16, 1, False)
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        common_trans.append(normalize)

        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        if 'train' in splits:
            train_set = d_func(data_root, train=True, transform=train_compose, download=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if 'val' in splits or 'test' in splits:
            test_set = d_func(data_root, train=False, transform=test_compose)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)
            val_loader = test_loader

    elif data.find('tiny') >= 0:
        print('loading ' + data)
        print(config_of_data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if config_of_data['augmentation']:
            print('with data augmentation')
            aug_trans = [
                transforms.RandomHorizontalFlip(),
            ]
        else:
            aug_trans = []
        common_trans = [transforms.ToTensor()]
        common_trans.append(normalize)
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        train_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/train', transform=train_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        val_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/val', transform=test_compose)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        test_set = dset.ImageFolder('/home/vslab2018/data/tiny-imagenet-200/test', transform=test_compose)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    elif data.find('ImgNet') >= 0:
        print('loading ' + data)
        print(config_of_data)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = dset.ImageFolder('/mnt/work/data/data/raw-data/train', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        test_set = dset.ImageFolder('/mnt/work/data/data/raw-data/val', transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        val_loader = test_loader

    else:
        raise NotImplemented

    return train_loader, val_loader, test_loader
