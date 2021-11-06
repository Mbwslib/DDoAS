import torch
from torch import nn
import torchvision
from CircConv import AttSnake
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDoM(nn.Module):

    def __init__(self):

        super(DDoM, self).__init__()

        self.fc0 = nn.Linear(15, 1024)

        self.sig = nn.Sigmoid()

        self.fc1 = nn.Linear(1024, 2048)

        self.avg_pool2 = nn.AvgPool2d(7, stride=1)
        self.fc2 = nn.Linear(3072, 1024)

        self.avg_pool3 = nn.AvgPool2d(14, stride=1)
        self.fc3 = nn.Linear(2048, 512)

        self.avg_pool4 = nn.AvgPool2d(28, stride=1)
        self.fc4 = nn.Linear(1536, 256)

    def init_weights(self):
        self.fc0.weight.data.normal_(0, 0.01)
        self.fc0.bias.data.zero_()
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.normal_(0, 0.01)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.normal_(0, 0.01)
        self.fc4.bias.data.zero_()

    def forward(self, cls_feature, o4, o3, o2, o1):

        bs = o4.size(0)

        gs = self.fc0(cls_feature)

        gs1 = torch.bernoulli(self.sig(self.fc1(gs)))
        o4 = o4.view(bs, 2048, -1)
        gs1 = gs1.unsqueeze(2).expand_as(o4)
        doo4 = o4.mul(gs1).view(bs, 2048, 7, 7)

        gs2 = self.avg_pool2(doo4).view(bs, -1)
        gs2 = torch.bernoulli(self.sig(self.fc2(torch.cat([gs, gs2], dim=1))))
        o3 = o3.view(bs, 1024, -1)
        gs2 = gs2.unsqueeze(2).expand_as(o3)
        doo3 = o3.mul(gs2).view(bs, 1024, 14, 14)

        gs3 = self.avg_pool3(doo3).view(bs, -1)
        gs3 = torch.bernoulli(self.sig(self.fc3(torch.cat([gs, gs3], dim=1))))
        o2 = o2.view(bs, 512, -1)
        gs3 = gs3.unsqueeze(2).expand_as(o2)
        doo2 = o2.mul(gs3).view(bs, 512, 28, 28)

        gs4 = self.avg_pool4(doo2).view(bs, -1)
        gs4 = torch.bernoulli(self.sig(self.fc4(torch.cat([gs, gs4], dim=1))))
        o1 = o1.view(bs, 256, -1)
        gs4 = gs4.unsqueeze(2).expand_as(o1)
        doo1 = o1.mul(gs4).view(bs, 256, 56, 56)

        return doo4, doo3, doo2, doo1


class DDoAS(nn.Module):

    def __init__(self, num_classes):

        '''
        Dense De-overlap Attention Snake for real-time prohibited item segmentation
        '''

        super(DDoAS, self).__init__()

        #  load pretrained model (we take ResNet-50 as an example)
        resnet = torchvision.models.resnet50(pretrained=False)
        resnet.load_state_dict(torch.load('./checkpoint/resnet50-19c8e357.pth'))

        for item in resnet.children():
            if isinstance(item, nn.BatchNorm2d):
                item.affine = False

        self.base_features = nn.Sequential(resnet.conv1,
                                           resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool)  # (bs, 64, 56, 56)
        self.res1 = resnet.layer1  # (bs, 256, 56, 56)
        self.res2 = resnet.layer2  # (bs, 512, 28, 28)
        self.res3 = resnet.layer3  # (bs, 1024, 14, 14)
        self.res4 = resnet.layer4  # (bs, 2048, 7, 7)

        # set Dense De-overlap Module

        self.ddom = DDoM()

        # conv layer for enhance features
        self.enconv4 = nn.Sequential(nn.Conv2d(2048, 256, 7, 1, 3), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 7, 1, 3), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 7, 1, 3), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))

        self.enconv3 = nn.Sequential(nn.Conv2d(1024, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))

        self.enconv2 = nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 5, 1, 2), nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.enconv1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))

        self.enff = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))

        # linear layer for classification
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.cls_spv = nn.Linear(2048, num_classes)

        # edge supervision
        self.edge_spv = nn.Conv2d(128, 1, 3, 1, 1)

        # generate weight map
        self.att_F = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 1, 3, 1, 1), nn.ReLU(inplace=True),
                                   nn.Sigmoid())

        # set Attention Deforming Module
        self.attsnake = AttSnake(n_adj=2)

        self.prediction = nn.Sequential(nn.Conv1d(128, 64, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(64, 64, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(64, 2, 1))

        # Initializes some parameters for easier convergence
        self.init_weights()

    def get_bilinear_interpolation(self, ff, Fatt, img_poly):

        '''
        Extract each vertex features by using bilinear interpolation
        '''

        img_poly = img_poly.clone()
        img_poly[..., 0] = img_poly[..., 0] / 56. - 1
        img_poly[..., 1] = img_poly[..., 1] / 56. - 1

        bs = ff.size(0)
        gcn_feature = torch.zeros([bs, ff.size(1), img_poly.size(1)]).to(device)  # (bs, 128, point_num)
        pw = torch.zeros([bs, 1, img_poly.size(1)]).to(device)

        for i in range(bs):
            grid = img_poly[i:i + 1].unsqueeze(1)
            bilinear_feature = torch.nn.functional.grid_sample(ff[i:i + 1], grid=grid, align_corners=True)[0].permute(1, 0, 2)
            gcn_feature[i] = bilinear_feature

            fatt_feature = torch.nn.functional.grid_sample(Fatt[i:i + 1], grid=grid, align_corners=True)[0].permute(1, 0, 2)
            pw[i] = fatt_feature


        #point_center = (torch.min(img_poly, dim=1)[0] + torch.max(img_poly, dim=1)[0]) * 0.5
        #point_center = point_center[:, None]
        #ct_feature = torch.zeros([batch_size, concat_feature.size(1), point_center.size(1)]).to(device)  # (bs, 128, 1)

        #for j in range(batch_size):
        #    grid = point_center[j:j + 1].unsqueeze(1)
        #    ct_bilinear_feature = torch.nn.functional.grid_sample(concat_feature[j:j + 1], grid=grid)[0].permute(1, 0, 2)
        #    ct_feature[j] = ct_bilinear_feature

        #fuse_feature = torch.cat([gcn_feature, ct_feature.expand_as(gcn_feature)], dim=1)
        #fuse_feature = self.fuse(fuse_feature)

        return gcn_feature, pw

    def normalize_poly(self, img_poly):

        mi = torch.min(img_poly, dim=1, keepdim=True)[0].expand_as(img_poly)
        ma = torch.max(img_poly, dim=1, keepdim=True)[0].expand_as(img_poly)

        new_poly = (img_poly - mi) / (ma - mi)

        return new_poly, mi, ma


    def init_weights(self):
        self.cls_spv.weight.data.normal_(0, 0.01)
        self.cls_spv.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for name, param in self.named_parameters():
            if 'fc' in name and 'weight' in name:
                nn.init.normal_(param, 0.0, 0.01)
            if 'fc' in name and 'bias' in name:
                nn.init.constant_(param, 0.0)
            if ('enconv' or 'enff' or 'edge_spv') in name and 'weight' in name:
                nn.init.normal_(param, 0.0, 0.01)
            if ('enconv' or 'enff' or 'edge_spv') in name and 'bias' in name:
                nn.init.constant_(param, 0.0)


    def forward(self, image, img_poly):

        image = self.base_features(image)
        o1 = self.res1(image)  # (bs, 256, 56, 56)
        o2 = self.res2(o1)     # (bs, 512, 28, 28)
        o3 = self.res3(o2)     # (bs, 1024, 14, 14)
        o4 = self.res4(o3)     # (bs, 2048, 7, 7)

        # classification branches
        cls_feature = self.avg_pool(o4).view(o4.size(0), -1)
        cls_feature = self.cls_spv(cls_feature)

        # DDoM
        doo4, doo3, doo2, doo1 = self.ddom(cls_feature, o4, o3, o2, o1)

        # O2OFM
        doo1 = self.enconv1(doo1)
        fo4 = self.enconv4(doo4) + doo1
        fo3 = self.enconv3(doo3) + doo1
        fo2 = self.enconv2(doo2) + doo1
        ff = self.enff(fo2 + fo3 + fo4)  # (bs, 128, 56, 56)

        # edge supervision branches
        edge_feature = self.edge_spv(ff)  # (bs, 1, 56, 56)

        Fatt = self.att_F(ff)

        #  Deformation branches
        bilinear_feature, pw = self.get_bilinear_interpolation(ff, Fatt, img_poly)  # (bs, 128, point_num) (bs, 1, point_num)
        Fatt_bf = bilinear_feature.mul(pw)  # + bilinear_feature
        img_poly, mi, ma = self.normalize_poly(img_poly)
        app_features = torch.cat([Fatt_bf, img_poly.permute(0, 2, 1)], dim=1)  # (bs, 130, point_num)
        #app_features = torch.cat([bilinear_feature, img_poly.permute(0, 2, 1)], dim=1)  # (bs, 130, point_num)


        snake_feature = self.snake(app_features)  # (bs, 640, point_num)

        predict_offset = self.prediction(snake_feature).permute(0, 2, 1)  # (bs, point_num, 2)

        predict_offset = predict_offset * (ma - mi)

        return cls_feature, edge_feature, predict_offset

if __name__ == '__main__':

    from utils import *
    net = DDoAS(15)
    input_ellipse = get_ic(scale=4.)
    flops, params = profile(net, (torch.randn(1, 3, 224, 224), input_ellipse.unsqueeze(0),))
    print(flops)
    print(params)

