from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np


class Recorder:
    def __init__(self):
        self.metrics = {}
        self.kvs ={}

    def record(self, name, value):
        """
        Description:
            Using dictionary to record metrics. The value of the items is a list.
        Params:
            name -- key of the item to be saved, a string
            value -- value of the item to be saved, a number
        """
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        """
        Description:
            Get the mean of each metirc, which is used after an epoch.
        """
        self.kvs = {}
        for key in self.metrics.keys():
            self.kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return self.kvs
    
    def reset(self):
        """
        Description:
            reset the record, it will be used for every epoch.
        """
        del self.metrics
        del self.kvs
        self.metrics = {}
        self.kvs = {}


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(os.path.join(args.log_dir, args.model, '{id:02d}'.format(id=args.exp_id) + args.save_suffix))
        self.recorder = Recorder()
        self.model_dir = args.model_dir # the directory to save the model
        self.args = args

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.detach().cpu().numpy()

    def record_scalar(self, name, value):
        """
        Description:
            Using dictionary to record metrics of the current training step.
        Params:
            name -- key of the item to be saved, a string
            value -- value of the item to be saved, a scalar
        """
        self.recorder.record(name, value)

    def save_curves(self, epoch):
        """
        Description:
            save the mean value of each metric after training for an epoch
        Params:
            epoch -- the current epoch
        """
        kvs = self.recorder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)

    def save_imgs(self, names2imgs, epoch):
        """
        Description:
            save the image of result after training for an epoch
        Params:
            epoch -- the current epoch
            name_2imgs -- a dictionary with image_name(string) as the key
                          and the image(tensor) to be saved as the value
        """
        for name in names2imgs.keys():
            self.writer.add_image(name, names2imgs[name], epoch)

    def save_checkpoint(self, state_dict, epoch, step=0, is_best=False):
        """
        Description:
            save checkpoint of the model. It will save the state dict of the current model
        Params:
            state_dict -- the things need to be saved
            model_name -- the name of the model, such as atturesnext_feat, the checkpoint is saved according to model_name
            epoch -- current epoch
            step -- current step of the current epoch. Maybe you will not save the model
                    after the whole training epoch, but save the model at some middle point of the current epoch. 
                    e.g., you can save checkpoint every 1000 steps or so.
                    if it is set to 0, then not use it in the path
            is_best -- whether the model has the best evaluation results or not
        """
        # create model_name
        if step != 0:
            model_name = self.args.model + '_{epoch:03d}_{step:05d}.pth.tar'.format(epoch=epoch, step=step)
        else:
            model_name = self.args.model + '_{epoch:03d}.pth.tar'.format(epoch=epoch)
        # save model
        if is_best: # best model
            print('get the best model in epoch:' + str(epoch))
            model_dir = os.path.join(self.args.model_dir, 'best')
            os.system('rm -rf ' + self.args.model_dir +'/best/*') # remove the old best
            path = os.path.join(model_dir, model_name)
        else: # checkpoints
            path = os.path.join(self.args.model_dir, model_name)
        # don't save model, which depends on python path
        # save model state dict
        torch.save(state_dict, path)
        
