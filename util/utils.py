import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from util.autoaugment import CIFAR10Policy, SVHNPolicy
from util.criterions import LabelSmoothingCrossEntropyLoss
from util.da import RandomCropPaste

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit':
        from networks.vit import ViT
        breakpoint()
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    
    # JINLOVESPHO
    elif args.model_name == 'vit_parallel':
        from networks.vit_parallel import ViT_Parallel
        breakpoint()
        net = ViT_Parallel(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )  
        
    # JINLOVESPHO
    elif args.model_name == 'vit_gyu':
        from networks.vit_gyu import ViT_Gyu
        breakpoint()
        net = ViT_Gyu(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    
    elif args.model_name == 'vit-Tiny_crossVit':
        from networks.vit_tiny_crossvit import ViT_Tiny_CrossVit
        breakpoint()
        net = ViT_Tiny_CrossVit(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
    
    elif args.model_name == 'vit-Small_crossVit':
        from networks.vit_small_crossvit import ViT_Small_CrossVit
        breakpoint()
        net = ViT_Small_CrossVit(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )        
    
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    data_path = args.data_path
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(data_path, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(data_path, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(data_path, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(data_path, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.experiment_memo}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
