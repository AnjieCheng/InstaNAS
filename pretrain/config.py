# this is used for storing configurations of datasets & models

datasets = {
    'cifar10': {
        'num_classes': 10,
        'augmentation': False,
    },
    'cifar10+': {
        'num_classes': 10,
        'augmentation': True,
    },
    'cifar100': {
        'num_classes': 100,
        'augmentation': False,
    },
    'cifar100+': {
        'num_classes': 100,
        'augmentation': True,
    },
    'tiny': {
        'num_classes': 200,
        'augmentation': True,
    },
    'ImgNet': {
        'num_classes': 1000,
        'augmentation': True,
    },
}