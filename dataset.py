import glob
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import get_ic
import torchvision.transforms as transforms


prohibited_item_classes = {'Gun': 0, 'Knife': 1, 'Wrench': 2, 'Pliers': 3, 'Scissors': 4, 'Lighter': 5, 'Battery': 6,
                           'Bat': 7, 'Razor_blade': 8, 'Saw_blade': 9, 'Fireworks': 10, 'Hammer': 11,
                           'Screwdriver': 12, 'Dart': 13, 'Pressure_vessel': 14}
#cityscapes_classes = {'person': 0, 'car': 1, 'truck': 2, 'bicycle': 3, 'motorcycle': 4, 'rider': 5,
#                      'bus': 6, 'train': 7}

trans = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.838, 0.855, 0.784],
                                 std=[0.268, 0.225, 0.291])

class data_loader(Dataset):
    def __init__(self, split):

        self.split = split
        assert self.split in {'trainset', 'valset', 'testset'}

        self.dataset_size = len(glob.glob('{}/*.json'.format(split)))

    def __getitem__(self, ind):
        img_path = '{}/{}.png'.format(self.split, ind)
        edge_path = '{}/{}b.png'.format(self.split, ind)
        gt_path = '{}/{}.json'.format(self.split, ind)

        # Open json file where ground-truth are stored
        with open(gt_path, 'r', encoding='utf8', errors='ignore') as j:
            gt = json.load(j)
        label_ind = torch.LongTensor([prohibited_item_classes[gt['label']]])
        gt_ellipse = torch.FloatTensor(gt['polygon'])

        img = Image.open(img_path)
        img_edge = Image.open(edge_path)

        # Get the initial contour
        input_ellipse = get_ic(scale=4.)

        # PyTorch transformation pipeline for the image (totensor, normalizing, etc.)
        img = normalize(trans(img))
        img_edge = trans(img_edge)

        return img, img_edge, input_ellipse, label_ind, gt_ellipse

    def __len__(self):

        return self.dataset_size

class data_loader_test(data_loader):
    def __init__(self, split):

        self.split = split
        self.dataset_size = len(glob.glob('{}/*.json'.format(split)))

    def __getitem__(self, ind):
        img_path = '{}/{}.png'.format(self.split, ind)
        gt_path = '{}/{}.json'.format(self.split, ind)

        # Open json file where ground-truth are stored
        with open(gt_path, 'r', encoding='utf8', errors='ignore') as j:
            gt = json.load(j)
        label_ind = torch.LongTensor([prohibited_item_classes[gt['label']]])
        gt_ellipse = torch.FloatTensor(gt['polygon'])

        img = Image.open(img_path)

        # Get the initial contour
        input_ellipse = get_ic(scale=4.)

        # PyTorch transformation pipeline for the image (totensor, normalizing, etc.)
        img = normalize(trans(img))

        return img, input_ellipse, label_ind, gt_ellipse
