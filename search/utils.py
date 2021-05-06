import shutil
import os, sys, time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
import math

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt', 'w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(2).sum(1).sum(0)/policies.size(0)

    policy_set = [np.reshape(p.cpu().numpy().astype(
        np.int).astype(np.str), (-1)) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, sparsity, policy_set


def adjust_learning_rate_cos(optimizer, epoch, max_epochs, lr, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = max_epochs * nBatch
        T_cur = (epoch % max_epochs) * nBatch + batch
        new_lr = 0.5 * lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        new_lr, decay_rate = lr, 0.1
        if epoch >= max_epochs * 0.75:
            new_lr *= decay_rate**2
        elif epoch >= max_epochs * 0.5:
            new_lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch % self.epoch_step == 0:
                print(' [*] setting learning_rate to %.2E' % lr)



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


def get_transforms(instanet, dset):

    if dset == 'C10':
        mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std = [x/255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            cutout(16, 1, False),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif dset=='C100' :
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            cutout(16, 1, False),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

    elif dset == 'Tiny':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif dset == 'ImgNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test


def get_dataset(model, root='../data/'):

    instanet, dset = model.split('_')
    transform_train, transform_test = get_transforms(instanet, dset)

    if dset == 'C10':
        trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dset == 'C100':
        trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dset == 'ImgNet':
        trainset = torchdata.ImageFolder('/mnt/work/data/data/raw-data/train', transform_train)
        testset = torchdata.ImageFolder('/mnt/work/data/data/raw-data/val', transform_test)
    elif dset == 'Tiny':
        trainset = torchdata.ImageFolder('/home/anjie/Workspace/data/tiny-imagenet-200/train', transform=transform_train)
        testset = torchdata.ImageFolder('/home/anjie/Workspace/data/tiny-imagenet-200/val', transform=transform_test)
        print(testset)

    return trainset, testset


def get_model(model):

    from models import controller, base, instanas

    if model == 'InstaMobile_C10':
        instanet_checkpoint = '../pretrain/save/C10+-InstaNas+-Pretrain/checkpoint.pth.tar'
        instanet = instanas.MobileNet()
        agent = controller.Policy32([1, 1, 1], num_blocks=17, num_of_actions=5)

    elif model == 'InstaMobile_ImgNet':
        instanet_checkpoint = '../pretrain/save/ImgNet+-InstaNas+-Pretrain/checkpoint.pth.tar'
        instanet = instanas.MobileNet_224(num_classes=1000) #Special taylor for TinyImageNet
        agent = controller.Policy224([1,1,1,1], num_blocks=17, num_of_actions=5)

    else:
        raise NotImplementedError(' [*] Unkown model.')
    instanet = torch.nn.DataParallel(instanet).cuda()
    # load pretrained weights into meta-graph
    if instanet_checkpoint:
        instanet_checkpoint = torch.load(instanet_checkpoint)
        new_state = instanet.state_dict()
        new_state.update(instanet_checkpoint['state_dict'])
        instanet.load_state_dict(new_state)
    # print(instanet_checkpoint['state_dict'].keys())
    # instanet = torch.nn.DataParallel(instanet).cuda()
        # instanet.load_state_dict(instanet_checkpoint['state_dict'])
    return instanet, agent
