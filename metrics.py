"""
containing loss and metrics to evaluate the model
it was computed by using the pixels of the whold datasets!!!
Description in -- https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
"""
import torch

def get_precision_recall(cm):
    true_pos = torch.diag(cm)
    false_pos = torch.sum(cm, axis=0) - true_pos
    false_neg = torch.sum(cm, axis=1) - true_pos

    # precision = torch.mean(true_pos / (true_pos + false_pos))
    # recall = torch.mean(true_pos / (true_pos + false_neg))
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return precision, recall

def MIoU(pred, gt):
    """
    Description:
        Mean Intersection over Union
    Params:
        pred -- the predicted label, tensor of shape (N,H,W,C)
        gt -- the ground_truth label, tensor of shape (N,H,W,C)
    Tips:
        gt needs to be transformed to one_hot form
    """
    intersection = torch.sum(pred * gt, (1,2,3))
    union = torch.sum(pred, (1,2,3)) + torch.sum(gt, (1,2,3)) - intersection
    miou = torch.mean(intersection / union)

    return miou

def F1(pred, gt):
    """
    Description:
        Dice Coefficient (F1 Score)
    Params:
        pred -- the predicted label, tensor of shape (N,H,W,C)
        gt -- the ground_truth label, tensor of shape (N,H,W,C)
    Tips:
        gt needs to be transformed to one_hot form
    """
    intersection = torch.sum(pred * gt, (1,2,3))
    union = torch.sum(pred, (1,2,3)) + torch.sum(gt, (1,2,3))
    f1 = torch.mean(2 * intersection / union)

    return f1

def OA(pred, gt):
    """
    Description:
        OA, overall accuracy (pixel)
    Params:
        pred -- the predicted label, tensor of shape (N,H,W,C)
        gt -- the ground_truth label, tensor of shape (N,H,W,C)
    Tips:
        gt needs to be transformed to one_hot form
    """
    correct = torch.sum(pred * gt)
    all = torch.sum(gt)
    oa = correct / all

    return oa

if __name__ == '__main__':
    pred = torch.ones((5,3,6,6))
    gt = torch.ones((5,3,6,6))
    # x = OA(pred, gt)
