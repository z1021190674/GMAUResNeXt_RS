import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import math

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts



from options import get_checkpoint_path, prepare_train_args
from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_freeze_model, select_model, equip_multi_gpu
from utils.logger import Logger
from utils.torch_utils import freeze_weight, get_weight_dict, load_match_dict
from utils.viz import label2rgb, label2color
from metrics import  OA

class Trainer():
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)
        # seed
        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)  #为CPU设置种子用于生成随机数，以使得结果是确定的   　　 
            torch.cuda.manual_seed_all(args.seed) #为所有GPU设置随机种子；  　
            # may cause the network running slow??? -- https://zhuanlan.zhihu.com/p/141063432　  
            # torch.backends.cudnn.deterministic = True
        
        # dataloader
        if args.dataset == 'Potsdam':
            self.train_loader_400 = select_train_loader(args, res=400)
            self.train_loader_800 = select_train_loader(args, res=800)
            self.train_loader_1200 = select_train_loader(args, res=1200)
        elif args.dataset == 'Gid':
            self.train_loader = select_train_loader(args, res=args.res)

        self.val_loader = select_eval_loader(args)
        
        # model
        self.model = select_model(args)

        # load checkpoint -- 注意，如果想要使用.cuda()方法来将model移到GPU中，一定要确保这一步在构造Optimizer之前。因为调用.cuda()之后，model里面的参数已经不是之前的参数了。        
        if self.args.load_model_path != '': # check whether it needs to load checkpoint
            state_dict = torch.load(self.args.load_model_path)
            # model
            if self.args.load_not_strict:
                load_match_dict(self.model, state_dict['state_dict'])
            else:
                self.model.load_state_dict(state_dict['state_dict'].state_dict())
            # freeze weights
            if self.args.is_freeze:
                model = select_freeze_model(args)
                freeze_dict = get_weight_dict(model)
                # freeze_weight(self.model, freeze_dict) # not detach the backprop process
                # parallel training
                self.model = equip_multi_gpu(self.model, args)
                self.optimizer = torch.optim.Adam(filter(lambda p: p not in freeze_dict, self.model.parameters()), self.args.lr,
                                            betas=(self.args.momentum, self.args.beta),
                                            weight_decay=self.args.weight_decay)
            else:
                # parallel training
                self.model = equip_multi_gpu(self.model, args)
                param_list1, param_list2 = self.get_optim_param()
                # optimizer -- just updating the weights with requires_grad=True
                if self.args.is_pretrained:
                    self.optimizer = torch.optim.Adam([
                                                        {"params": param_list1, 'lr': self.args.lr*0.1},
                                                        {"params": param_list2, 'lr': self.args.lr},
                                                    ],                                                          # use different lr, lower learning rate for
                                                    betas=(self.args.momentum, self.args.beta),
                                                    weight_decay=self.args.weight_decay)
                else:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,                                                          # use different lr, lower learning rate for
                                                    betas=(self.args.momentum, self.args.beta),
                                                    weight_decay=self.args.weight_decay)
            
            if args.is_pretrain:  # for pretrain
                print("=> using pre-trained weights of " + args.load_model_path)
            else: # load checkpoint
                print("=> checkpoints of"  + args.load_model_path)
                self.optimizer.load_state_dict(state_dict['optimizer_dict'])
                # other settings
                self.args.current_epoch = state_dict['current_epoch']
                self.val_min_loss =state_dict['val_min_loss']
                self.val_max_oa = state_dict['val_max_oa']
        else:
            param_list1, param_list2 = self.get_optim_param()
            self.model = equip_multi_gpu(self.model, args)
            if self.args.is_pretrained:
                self.optimizer = torch.optim.Adam([
                                                    {"params": param_list1, 'lr': self.args.lr*0.1},
                                                    {"params": param_list2, 'lr': self.args.lr},
                                                ],                                                          # use different lr, lower learning rate for
                                                betas=(self.args.momentum, self.args.beta),
                                                weight_decay=self.args.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,                                                          # use different lr, lower learning rate for
                                                betas=(self.args.momentum, self.args.beta),
                                                weight_decay=self.args.weight_decay)
        # scheduler
        self.select_lr_scheduler()
        

    def train(self):
        for epoch in range(self.args.current_epoch ,self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            # save the metrics and loss in the tensorboard
            self.logger.save_curves(epoch)
            # dict for saving
            val_loss = self.logger.recorder.kvs['val/' + self.args.loss +' loss']
            val_oa = self.logger.recorder.kvs['val/' + 'OA']
            if epoch == 0 or self.args.is_pretrain:
                self.args.is_pretrain = False # it is done using args.is_pretrain
                self.val_min_loss = val_loss
                self.val_max_oa = val_oa
            state_dict = {
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer.state_dict(),
                'current_epoch': epoch + 1,
                'val_min_loss': self.val_min_loss,
                'val_max_oa': self.val_max_oa
            }
            
            # save best model
            # if val_loss < self.val_min_loss:
            #     self.val_min_loss = val_loss
            #     state_dict['val_min_loss'] = self.val_min_loss
            #     self.logger.save_checkpoint(state_dict, epoch, is_best=True)           
            if val_oa > self.val_max_oa:
                self.val_max_oa = val_oa
                state_dict['val_max_oa'] = self.val_max_oa
                self.logger.save_checkpoint(state_dict, epoch, is_best=True)

            # save model every self.args.model_epoch
            if (epoch + 1) % self.args.model_epoch == 0 and epoch != 0:
                self.logger.save_checkpoint(state_dict, epoch)

            # adjust learning rate
            self.scheduler.step(val_loss)
            # reset the recorder
            self.logger.recorder.reset()
    
    def train_per_epoch(self, epoch):
        # select train loader
        if self.args.dataset == 'Potsdam':
            self.train_loader = self.train_loader_800
            # if epoch % 3 == 0:
            #     self.train_loader = self.train_loader_400
            # elif epoch % 3 == 1:
            #     self.train_loader = self.train_loader_800
            # else:
            #     self.train_loader = self.train_loader_1200
        elif self.args.dataset == 'Gid':
            # if epoch % 1 == 0:
            #     self.train_loader = self.train_loader_1200
            pass


        # switch to train mode
        self.model.train()
        with tqdm.tqdm(self.train_loader, unit="batch") as train_loader:
            for i, data in enumerate(train_loader):
        # for i, data in enumerate(self.train_loader):
                img, label, prob = self.step(data)

                # compute loss
                metrics = self.compute_metrics(prob, label, is_train=True)

                # get the item for backward
                loss = metrics['train/' + self.args.loss + ' loss']

                # compute gradient and do Adam step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()                

                # logger record, save it after an epoch
                for key in metrics.keys():
                    self.logger.record_scalar(key, metrics[key].item())

                with torch.no_grad():
                    # only save img at last step
                    if i == len(self.train_loader) - 1:
                        pred = torch.argmax(prob, dim=1)
                        self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, is_train=True), epoch)

                # monitor training progress
                # information showed in the progress bar
                train_loader.set_description("Exp:" + self.args.model_dir + ", Epoch:{} in training".format(epoch))
                train_loader.set_postfix_str("loss_per_data:{:.5f}"
                                    .format(loss.item()))
                # if i % self.args.print_freq == 0:
                #     print('\nTrain: Epoch {} batch {} Loss {}'.format(epoch, i, loss))

    def val_per_epoch(self, epoch):
        # switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(self.val_loader, unit="batch") as val_loader:
                for i, data in enumerate(val_loader):
                    img, label, prob = self.step(data)
                    metrics = self.compute_metrics(prob, label, is_train=False)

                    for key in metrics.keys():
                        self.logger.record_scalar(key, metrics[key].item())

                    with torch.no_grad():
                        if i == len(self.val_loader) - 1:
                            pred = torch.argmax(prob, dim=1)
                            self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)
                
                    # monitor validation progress
                    # information showed in the progress bar
                    val_loader.set_description("Epoch:{} in validation".format(epoch))
    

    def step(self, data):
        """
        Description:
            forward prop of the training
        Params:
            data -- one batch from the dataset
        Return:
            img -- images in the gpu, tensor of shape(N,C,H,W)
            label -- labels in the gpu, tensor of shape(N,H,W)
            prob -- the result of the model
        """
        img, label = data
        # warp input
        img = img.cuda()
        label = label.cuda()

        # compute output
        prob = self.model(img)
        return img, label, prob

    def compute_metrics(self, prob, gt, is_train):
        """
        Description:
            Compute the loss and save as a dict named metrics,which is used in the tensorboard        
        """
        # you can call functions in metrics.py
        loss = self.compute_loss(prob, gt)
        pred = torch.argmax(prob, dim=1)
        with torch.no_grad():
            gt_one_hot = F.one_hot(gt, num_classes=self.args.num_classes)
            pred_one_hot = F.one_hot(pred, num_classes=self.args.num_classes)
            # evaluation metrics
            # miou = MIoU(pred_one_hot, gt_one_hot)
            # f1 = F1(pred_one_hot, gt_one_hot)
            oa = OA(pred_one_hot, gt_one_hot)
        
        # this is used for recording in tensorboard
        prefix = 'train/' if is_train else 'val/'
        metrics = {
            prefix + self.args.loss + ' loss': loss,
            # prefix + 'MIoU': miou,
            # prefix + 'F1': f1,
            prefix + 'OA': oa,
        }
        return metrics

    def compute_loss(self, prob, gt):
        """
        Description:
            compute loss according to args.loss
        """
        if self.args.loss == 'l1':
            loss = (prob - gt).abs().mean()
        elif self.args.loss == 'ce':
            # the output of the network is the probability
            loss = torch.nn.functional.nll_loss(prob, gt)
        # else:
        #     loss = torch.nn.functional.mse_loss(pred, gt)

        return loss

    def select_lr_scheduler(self): # select lr_scheduler according to self.args.lr_shceduler
        type2scheduler = {
            'ReduceLROnPlateau': ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.ropfactor, patience=self.args.patience),
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.T_0, T_mult=self.args.T_mult),
            'CosineOneCircle': torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: ((1 + math.cos(x * math.pi / self.args.epochs)) / 2) * (1 - self.args.lrf) + self.args.lrf),
        }
        self.scheduler = type2scheduler[self.args.lr_scheduler]

    
    def gen_imgs_to_write(self, img, pred, label, is_train):
        """
        Description:
            return a set of images to write, which are of one data
            you can override this method according to your visualization
        Params:
            img -- tensor of shape(C,H,W)
            pred -- tensor of shape(,H,W)
            label -- tensor of shape(H,W)
        Return:
            img -- numpy of shape(C,H,W)
            pred -- numpy of shape(3,H,W), which is the required input format as chw
            label -- numpy of shape(3,H,W), which is the required input format as chw
        """

        # this is used for recording in tensorboard
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': self.logger.tensor2img(img[0]).astype(np.uint8),
            # prefix + 'pred': np.transpose(label2rgb(self.logger.tensor2img(pred[0]).astype(np.uint8)), (2,0,1)),
            # prefix + 'label': np.transpose(label2rgb(np.transpose(self.logger.tensor2img(F.one_hot(label[0], num_classes=self.args.num_classes)).astype(np.uint8), (2,0,1))), (2,0,1)), 
            prefix + 'pred': np.transpose(label2color(self.logger.tensor2img(pred[0]).astype(np.uint8), self.args.dataset), (2,0,1)),  #(h,w,c) to (c,h,w) for summary writer
            prefix + 'label':  np.transpose(label2color(self.logger.tensor2img(label[0]).astype(np.uint8), self.args.dataset), (2,0,1)),  # (c,h,w) for summary writer
        }
    def get_optim_param(self,):
        param_list1 = []
        param_list2 = []
        for name, parms in self.model.named_parameters():
            if "encoder" in name or "first" in name:
                param_list1 += [parms]
            else:
                param_list2 += [parms]
        return param_list1, param_list2

    

if __name__ == "__main__":
    import os
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    trainer = Trainer()
    trainer.train()