import torch


def load_match_dict(model, pretrain_dict):
    """
    Description:
        load model, while filtering out unnecessary keys.
        It already handles the state dict of multi-gpu-training model.
        Not handling state dict with different keys that should be matched(to be completed).
    Params:
        pretrain_dict -- the state dict of the pretrained model
        model -- a subclass of the nn.Module
    """
    # model: single gpu model, please load dict before warp with nn.DataParallel
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def freeze_weight(model, freeze_dict):
    """
    Description:
        freeze the weight according to the given dict, and unfreeze the weight outside the dict
    Return:
        # model -- the model to 
    """
    for name, value in model.named_parameters():
        if name in freeze_dict:
            value.requires_grad = False
        else:
            value.requires_grad = True

def print_parameters(model):
    for name, value in model.named_parameters():
        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))

def get_weight_dict(model): # get the weight dict of the model
    weight_dict = {}
    for name, value in model.named_parameters():
        weight_dict[name] = value
    return weight_dict

