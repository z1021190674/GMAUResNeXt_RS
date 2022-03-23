# evaluation results saved in args.result_dir, which is depending on the selected model.
# When the model is checkpoints\UResNeXt_nlocal\0, then the result_dir will be
# checkpoints\UResNeXt_nlocal\0\UResNeXt_nlocal_000_test_info.

import torch
import torch.nn.functional as F
from torchvision import transforms
# from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import tqdm
import os

from data.data_entry import select_eval_loader
from model.model_entry import equip_multi_gpu, select_model
from options import prepare_test_args
from utils.logger import Recorder
from utils.viz import label2color, label2rgb
from metrics import  get_precision_recall
from metrics import  OA,F1,MIoU



class Evaluator:
    def __init__(self):
        args = prepare_test_args()
        self.args = args
        self.model = select_model(args)
        # need to set this before the first use of cuda rather than after that !!!
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        # self.model.load_state_dict(torch.load(args.load_model_path).state_dict()['state_dict'])
        if args.is_gpu:
            pretrain_dict = torch.load(args.load_model_path)['state_dict']
        else:
            pretrain_dict = torch.load(args.load_model_path, map_location='cpu')['state_dict']
        pretrain_dict = {k.replace('module.', ''): v for k, v in pretrain_dict.items()}
        self.model.load_state_dict(pretrain_dict)
        # model dataloader recorder
        self.model = equip_multi_gpu(self.model, args)
        self.model.eval()
        self.val_loader = select_eval_loader(args, is_eval=True)
        self.recorder = Recorder()
        # confusion matrix
        self.cm = torch.zeros((args.num_classes-1, args.num_classes-1)).cuda()
        # for test cm
        # self.cm = torch.zeros((args.num_classes, args.num_classes)).cuda()

    def eval(self):
        with torch.no_grad():
            with tqdm.tqdm(self.val_loader, unit="batch") as val_loader:
                for i, data in enumerate(val_loader):
                    img, label = data
                    if self.args.is_tta:
                        img, label, prob = self.step_tta(data)
                    else:
                        img, label, prob = self.step(data)


                    pred = torch.argmax(prob, dim=1)
                    cm = self.get_cm(pred, label)
                    self.cm = self.cm + cm

                    # visualization of the results
                    if i % self.args.viz_freq == 0:
                        pred = torch.argmax(prob, dim=1)
                        self.viz_per_batch(img, pred, label, i)

                    # monitor validation progress
                    # information showed in the progress bar
                    val_loader.set_description("Testing for " + self.args.model_dir)

            metrics = self.compute_metrics(self.cm)
            for key in metrics.keys():
                self.recorder.record(key, metrics[key])
            metrics = self.recorder.summary()
            result_txt_path = os.path.join(self.args.result_dir, 'result.txt')

            # write metrics to result dir,
            # you can also use pandas or other methods for better stats !!!
            
            cm_norm = self.get_cm_normal(self.cm).cpu().numpy()
            self.cm = self.cm.cpu().numpy()
            np.savetxt( os.path.join(self.args.result_dir,"cm.csv"), self.cm, delimiter=",")
            np.savetxt( os.path.join(self.args.result_dir,"cm_norm.csv"), cm_norm, delimiter=",")
            with open(result_txt_path, 'a') as fd:
                fd.write(str(metrics)+'\n')

    def compute_metrics(self, cm):
        """
        Description:
            Compute the metrics and save as a dict named metrics, which is used in the tensorboard and save in result.txt      
        """
        # for gid -- get rid of class 10 (shrub)
        col_1, col_2 = cm[:,0:9], cm[:,10:]
        temp = torch.cat((col_1,col_2),dim=1)
        row_1, row_2 = temp[0:9,:], temp[10:,:]
        cm = torch.cat((row_1,row_2),dim=0)

        precision, recall = get_precision_recall(cm)
        per_class_iou = torch.diag(cm) / (cm.sum(1) + cm.sum(0) - torch.diag(cm))  
        miou = torch.mean(per_class_iou)
        per_class_oa = torch.diag(cm) / cm.sum(1)
        oa = torch.mean(per_class_oa)
        f1_per_class = 2 * precision * recall /(precision + recall)
        f1 = torch.mean(f1_per_class)

        # this is used for recording in tensorboard

        metrics = {
            'MIoU': miou,
            'F1':f1,
            'OA': oa,
            'f1_per_class': f1_per_class,
        }
        return metrics

    # def compute_metrics(self, prob, gt):
    #     """
    #     Description:
    #         Compute the loss and save as a dict named metrics,which is used in the tensorboard        
    #     """
    #     # you can call functions in metrics.py
    #     loss = self.compute_loss(prob, gt)
    #     pred = torch.argmax(prob, dim=1)
    #     with torch.no_grad():
    #         gt_one_hot = F.one_hot(gt, num_classes=self.args.num_classes)
    #         pred_one_hot = F.one_hot(pred, num_classes=self.args.num_classes)
    #         # evaluation metrics
    #         miou = MIoU(pred_one_hot, gt_one_hot)
    #         f1 = F1(pred_one_hot, gt_one_hot)
    #         oa = OA(pred_one_hot, gt_one_hot)
        
    #     # this is used for recording in tensorboard
    #     metrics = {
    #         'MIoU': miou,
    #         'F1': f1,
    #         'OA': oa,
    #     }
    #     return metrics


    def viz_per_batch(self, img, pred, gt, step):
        # call functions in viz.py
        # here is an example about segmentation
        img_np = img[0].cpu().numpy().transpose((1, 2, 0))
        pred_np = label2color(pred[0].cpu().numpy(), dataset=self.args.dataset)
        gt_np = label2color(gt[0].cpu().numpy(), dataset=self.args.dataset)
        # arranged in a row
        viz = np.concatenate([img_np, pred_np, gt_np], axis=1)
        viz_path = os.path.join(self.args.result_dir, "%04d.jpg" % step)
        cv2.imwrite(viz_path, viz)
    
    def step_tta(self, data):
        img, label = data
        shape = img.shape[2]
        # warp input
        if self.args.is_gpu:
            img = img.cuda()
            label = label.cuda()
        # hvflip
        img_tmp = transforms.functional.hflip(img)
        img_tmp = transforms.functional.vflip(img_tmp)
        prob1 = self.model(img_tmp)
        prob1 =  transforms.functional.vflip(prob1)
        prob1 =  transforms.functional.hflip(prob1)
        # h flip
        img_tmp = transforms.functional.hflip(img)
        prob2 = self.model(img_tmp)
        prob2 =  transforms.functional.hflip(prob2)
        # v filp
        img_tmp = transforms.functional.vflip(img)
        prob3 = self.model(img_tmp)
        prob3 =  transforms.functional.vflip(prob3)
        # no flip
        prob4 = self.model(img)
        # compute output
        prob = (prob1 + prob2 + prob3 + prob4) / 4.0
        # crop to test
        img = img[:,:,int(shape/4):int(3*shape/4),int(shape/4):int(3*shape/4)]
        label = label[:,int(shape/4):int(3*shape/4),int(shape/4):int(3*shape/4)]
        prob = prob[:,:,int(shape/4):int(3*shape/4),int(shape/4):int(3*shape/4)]
        return img, label, prob

    def step(self, data):
        img, label = data
        # warp input
        img = img.cuda()
        label = label.cuda()

        # compute output
        prob = self.model(img)
        return img, label, prob
    
    def get_cm(self, pred, gt):
        """
        Description:
            get the confusion matrix of one batch
        Params:
            pred -- tensor of shape (n,h,w)
            gt -- tensor of shape (n,h,w)
        return:
            confusionMatrix -- tensor of shape(num_class, num_class)
        """
        # remove classes from unlabeled pixels in gt image and predict
        num_classes = self.args.num_classes - 1 # don't count the class - clutter/background
        # num_classes = self.args.num_classes

        mask_1 = (pred >= 0) & (pred < num_classes)
        mask_2 = (gt >= 0) & (gt < num_classes)
        mask = mask_1 & mask_2

        label = num_classes * gt[mask] + pred[mask]

        count = torch.bincount(label, minlength=num_classes**2)

        confusionMatrix = count.reshape(num_classes, num_classes)

        return confusionMatrix

    def get_cm_normal(self, cm): # the diagnol of the cm_normal is the precision
        return cm / torch.sum(cm,keepdim=True,dim=1)


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.eval()
    x=''
    # torch.load('checkpoints/UResNeXt/02/best/UResNeXt_044.pth.tar')