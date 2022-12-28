import numpy as np
import torch.nn as nn
import torch
import cv2

def get_ellipse(box, point_num):
    point_array = np.zeros(shape=(point_num, 2), dtype=np.float32)
    for i in range(point_num):
        theta = 1.0 * i / point_num * 2 * np.pi
        x = np.cos(theta)
        y = -np.sin(theta)
        point_array[i, 0] = x
        point_array[i, 1] = y
    point_array /= 2
    point_array += 0.5
    w, h = box[2] - box[0], box[3] - box[1]
    point_array *= np.array([w, h])
    point_array = point_array + np.array([box[0], box[1]])
    return point_array

def get_ic(scale=4.):

    point_array = get_ellipse((20, 40, 204, 184), 60)
    point_array = point_array.tolist()
    points = [point_array[0]] + list(reversed(point_array[1:]))
    points = torch.FloatTensor(points)
    points[..., 0] = points[..., 0] / scale
    points[..., 1] = points[..., 1] / scale

    return points

def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def clip_gradient(optimizer, grad_clip):

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_accuracy, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'model': model,
             'optimizer': optimizer,
             'val_accuracy': val_accuracy}

    filename = 'HR_checkpoint.pth.tar'
    torch.save(state, './checkpoint/' + filename)
    if is_best:
        torch.save(state, './checkpoint/BEST_' + filename)


def accuracy(new_ellipse, gt_ellipse):

    IoU = 0.
    batch_size = new_ellipse.size(0)
    for i in range(batch_size):
        new_ei = new_ellipse[i].tolist()
        gt_ei = gt_ellipse[i].tolist()
        iou, _ = iou_from_poly(new_ei, gt_ei, 224, 224)
        IoU += iou
    acc = IoU / batch_size

    return acc

def iou_from_mask(pred, gt):

    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union

    return iou, union

def iou_from_poly(pred, gt, height, width):

    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    masks[0] = draw_poly(masks[0], pred)

    masks[1] = draw_poly(masks[1], gt)

    return iou_from_mask(masks[0], masks[1])

def draw_poly(mask, poly):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, np.int32([poly]), 255)

    return mask

