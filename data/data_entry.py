from data.potsdam_dataset import Potsdam
from data.gid_dataset import Gid
from torch.utils.data import DataLoader


def get_dataset(args, res, is_train=True, is_eval=False):
    if args.dataset == 'Potsdam':
        dataset =  Potsdam(args, is_train, is_eval, res)
    elif args.dataset == 'Gid':
        dataset =  Gid(args, is_train, is_eval, res)
    return dataset


def select_train_loader(args, res=0):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset(args, res)
    print('{} samples found in train'.format(len(train_dataset)))
    if args.dataset == 'Potsdam':
        if res==400:
            train_loader = DataLoader(train_dataset, int(8), shuffle=True, num_workers=8, pin_memory=True)
        elif res==800:
            train_loader = DataLoader(train_dataset, int(4), shuffle=True, num_workers=8, pin_memory=True)
        elif res==1200:
            train_loader = DataLoader(train_dataset, int(1), shuffle=True, num_workers=8, pin_memory=True)
    elif args.dataset == 'Gid':
        if res==400:
            train_loader = DataLoader(train_dataset, int(16), shuffle=True, num_workers=8, pin_memory=True)
        elif res==600:
            train_loader = DataLoader(train_dataset, int(8), shuffle=True, num_workers=8, pin_memory=True)
        elif res==800:
            train_loader = DataLoader(train_dataset, int(4), shuffle=True, num_workers=8, pin_memory=True)
        elif res==1200:
            train_loader = DataLoader(train_dataset, int(2), shuffle=True, num_workers=8, pin_memory=True)
        elif res==640:
            train_loader = DataLoader(train_dataset, int(4), shuffle=True, num_workers=8, pin_memory=True)
        elif res==480:
            train_loader = DataLoader(train_dataset, int(8), shuffle=True, num_workers=8, pin_memory=True)
        elif res==320:
            train_loader = DataLoader(train_dataset, int(16), shuffle=True, num_workers=8, pin_memory=True)
    return train_loader


def select_eval_loader(args, is_eval=False):
    """
    is_eval -- True for test, False for validation
    """
    if args.dataset == 'Gid':
        eval_dataset = get_dataset(args, res=args.res, is_train=False, is_eval=is_eval)
        print('{} samples found in val'.format(len(eval_dataset)))
        val_loader = DataLoader(eval_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        eval_dataset = get_dataset(args, res=0, is_train=False, is_eval=is_eval) # res=0 for test and validation 
        print('{} samples found in val'.format(len(eval_dataset)))
        val_loader = DataLoader(eval_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader


