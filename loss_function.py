import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def smooth_l1(self, x):

        if torch.abs(x) < 1:
            y = 0.5 * (x ** 2)
        else:
            y = torch.abs(x) - 0.5

        return y

    def forward(self, cls_feature, label_ind, new_ellipse, gt_ellipse, edge_feature, img_edge):

        bs = new_ellipse.size(0)
        # loss_1 for classification
        metric_1 = nn.CrossEntropyLoss()
        l1 = metric_1(cls_feature, label_ind)

        # loss_2 for edge supervision
        l2 = F.binary_cross_entropy_with_logits(edge_feature, img_edge, reduction='sum') / bs

        # loss_3 for distance regression
        s = torch.pow(gt_ellipse - new_ellipse, 2)
        dist = torch.sqrt(s[:, :, 0] + s[:, :, 1] + 1e-10)
        l3 = 0.
        for i in range(bs):
            p = dist[i]
            for j in range(60):
                y = self.smooth_l1(p[j])
                l3 += y
        l3 = l3 / bs

        return l1, l2, l3

