import torch.backends.cudnn as cudnn
import torch.utils
import torchvision.transforms as transforms
from dataset import *
from utils import *
from loss_function import *

prohibited_item_classes = {'Gun': 0, 'Knife': 1, 'Wrench': 2, 'Pliers': 3, 'Scissors': 4, 'Lighter': 5, 'Battery': 6,
                           'Bat': 7, 'Razor_blade': 8, 'Saw_blade': 9, 'Fireworks': 10, 'Hammer': 11,
                           'Screwdriver': 12, 'Dart': 13, 'Pressure_vessel': 14}
cityscapes_classes = {'person': 0, 'car': 1, 'truck': 2, 'bicycle': 3, 'motorcycle': 4, 'rider': 5,
                      'bus': 6, 'train': 7}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint = './checkpoint/BEST_checkpoint.pth.tar'
batch_size = 1

# load model
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# normalization transform
trans = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.838, 0.855, 0.784],
                                 std=[0.268, 0.225, 0.291])

def test():

    test_loader = torch.utils.data.DataLoader(data_loader_test('testset'), batch_size=batch_size, shuffle=True)
    gun_iou = list()
    knife_iou = list()
    wrench_iou = list()
    pliers_iou = list()
    scissors_iou = list()
    lighter_iou = list()
    battery_iou = list()
    bat_iou = list()
    razor_blade_iou = list()
    saw_blade_iou = list()
    fireworks_iou = list()
    hammer_iou = list()
    screwdriver_iou = list()
    dart_iou = list()
    pressure_vessel_iou = list()
    #person_iou = list()
    #car_iou = list()
    #truck_iou = list()
    #bicycle_iou = list()
    #motorcycle_iou = list()
    #rider_iou = list()
    #bus_iou = list()
    #train_iou = list()


    for i, (img, input_ellipse, label_ind, gt_ellipse) in enumerate(test_loader):

        # move to GPU, if available
        img = img.to(device)
        input_ellipse = input_ellipse.to(device)
        label_ind = label_ind.to(device)
        gt_ellipse = gt_ellipse.to(device)

        cls_feature, edge_feature, predict_offset = model(img, input_ellipse)
        new_ellipse = input_ellipse + predict_offset
        new_ellipse[..., 0] = new_ellipse[..., 0] * 4.
        new_ellipse[..., 1] = new_ellipse[..., 1] * 4.

        # calculate IoU
        IoU = accuracy(new_ellipse, gt_ellipse)
        if label_ind == 0:
            gun_iou.append(IoU)
        elif label_ind == 1:
            knife_iou.append(IoU)
        elif label_ind == 2:
            wrench_iou.append(IoU)
        elif label_ind == 3:
            pliers_iou.append(IoU)
        elif label_ind == 4:
            scissors_iou.append(IoU)
        elif label_ind == 5:
            lighter_iou.append(IoU)
        elif label_ind == 6:
            battery_iou.append(IoU)
        elif label_ind == 7:
            bat_iou.append(IoU)
        elif label_ind == 8:
            razor_blade_iou.append(IoU)
        elif label_ind == 9:
            saw_blade_iou.append(IoU)
        elif label_ind == 10:
            fireworks_iou.append(IoU)
        elif label_ind == 11:
            hammer_iou.append(IoU)
        elif label_ind == 12:
            screwdriver_iou.append(IoU)
        elif label_ind == 13:
            dart_iou.append(IoU)
        elif label_ind == 14:
            pressure_vessel_iou.append(IoU)
        #if label_ind == 0:
        #    person_iou.append(IoU)
        #elif label_ind == 1:
        #    car_iou.append(IoU)
        #elif label_ind == 2:
        #    truck_iou.append(IoU)
        #elif label_ind == 3:
        #    bicycle_iou.append(IoU)
        #elif label_ind == 4:
        #    motorcycle_iou.append(IoU)
        #elif label_ind == 5:
        #    rider_iou.append(IoU)
        #elif label_ind == 6:
        #    bus_iou.append(IoU)
        #elif label_ind == 7:
        #    train_iou.append(IoU)


    print('Gun_IoU: {:.3f}\t'
          'Knife_IoU: {:.3f}\t'
          'Wrench_IoU: {:.3f}\t'
          'Pliers_IoU: {:.3f}\t'
          'Scissors_IoU: {:.3f}\t'
          'Lighter_IoU: {:.3f}\t'
          'Battery_IoU: {:.3f}\t'
          'Bat_IoU: {:.3f}\t'
          'Razor_blade_IoU: {:.3f}\t'
          'Saw_blade_IoU: {:.3f}\t'
          'Fireworks_IoU: {:.3f}\t'
          'Hammer_IoU: {:.3f}\t'
          'Screwdriver_IoU: {:.3f}\t'
          'Dart_IoU: {:.3f}\t'
          'Pressure_vessel_IoU: {:.3f}\t'.format(sum(gun_iou) / len(gun_iou),
                                                 sum(knife_iou) / len(knife_iou),
                                                 sum(wrench_iou) / len(wrench_iou),
                                                 sum(pliers_iou) / len(pliers_iou),
                                                 sum(scissors_iou) / len(scissors_iou),
                                                 sum(lighter_iou) / len(lighter_iou),
                                                 sum(battery_iou) / len(battery_iou),
                                                 sum(bat_iou) / len(bat_iou),
                                                 sum(razor_blade_iou) / len(razor_blade_iou),
                                                 sum(saw_blade_iou) / len(saw_blade_iou),
                                                 sum(fireworks_iou) / len(fireworks_iou),
                                                 sum(hammer_iou) / len(hammer_iou),
                                                 sum(screwdriver_iou) / len(screwdriver_iou),
                                                 sum(dart_iou) / len(dart_iou),
                                                 sum(pressure_vessel_iou) / len(pressure_vessel_iou)))
    #print('Person_IoU: {:.3f}\t'
    #      'Car_IoU: {:.3f}\t'
    #      'Truck_IoU: {:.3f}\t'
    #      'Bicycle_IoU: {:.3f}\t'
    #      'Motorcycle_IoU: {:.3f}\t'
    #      'Rider_IoU: {:.3f}\t'
    #      'Bus_IoU: {:.3f}\t'
    #      'Train_IoU: {:.3f}\t'.format(sum(person_iou) / len(person_iou),
    #                                   sum(car_iou) / len(car_iou),
    #                                   sum(truck_iou) / len(truck_iou),
    #                                   sum(bicycle_iou) / len(bicycle_iou),
    #                                   sum(motorcycle_iou) / len(motorcycle_iou),
    #                                   sum(rider_iou) / len(rider_iou),
    #                                   sum(bus_iou) / len(bus_iou),
    #                                   sum(train_iou) / len(train_iou)))
if __name__ == '__main__':
    test()
