import torch.utils
import torchvision.transforms as transforms
from models import DDoAS
from dataset import *
from utils import *
from loss_function import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
start_epoch = 0
epochs = 1000
grad_clip = 5.
learning_rate = 0.0001
epochs_since_improvement = 0
print_freq = 50  # 300
best_accuracy = 0.
checkpoint = None # './checkpoint/checkpoint.pth.tar'  # path to checkpoint
batch_size = 32
val_batch_size = 16


def main():

    '''
    Training and Validation
    '''

    global best_accuracy, epochs_since_improvement, checkpoint, start_epoch

    # initialize / load checkpoint
    if checkpoint is None:
        model = DDoAS(num_classes=15)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=learning_rate)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        best_accuracy = checkpoint['val_accuracy']


    # move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = loss().to(device)

    # custom dataloaders
    train_loader = torch.utils.data.DataLoader(data_loader('trainset'), batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(data_loader('valset'), batch_size=val_batch_size, shuffle=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        if epochs_since_improvement == 30:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 10 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(train_loader, model, criterion, optimizer, epoch)

        # One epoch's validation
        val_accuracy = validate(val_loader, model, criterion)

        # Check if there was an improvement
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(val_accuracy, best_accuracy)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0

        # save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_accuracy, is_best)

def train(train_loader, model, criterion, optimizer, epoch):

    '''
    performs one epoch's training
    '''

    model.train()

    train_loss = list()
    closs = list()
    eloss = list()
    ploss = list()
    train_acc = list()

    for i, (img, img_edge, input_ellipse, label_ind, gt_ellipse) in enumerate(train_loader):

        # move to GPU, if availble
        img = img.to(device)
        img_edge = img_edge.to(device)
        input_ellipse = input_ellipse.to(device)
        label_ind = label_ind.to(device)
        gt_ellipse = gt_ellipse.to(device)

        # forward prop.
        cls_feature, edge_feature, predict_offset = model(img, input_ellipse)  # (bs, class_num), (bs, point_num, 2)

        label_ind = label_ind.view(-1)  # (1, bs)

        # update contour
        new_ellipse = input_ellipse + predict_offset
        new_ellipse[..., 0] = new_ellipse[..., 0] * 4.
        new_ellipse[..., 1] = new_ellipse[..., 1] * 4.

        # calculate loss
        cls_loss, edge_loss, point_reg_loss = criterion(cls_feature, label_ind, new_ellipse, gt_ellipse, edge_feature, img_edge)

        loss_value = cls_loss + 0.01 * edge_loss + point_reg_loss * 0.01

        # Back prop.
        optimizer.zero_grad()
        loss_value.backward()

        train_loss.append(loss_value.item())
        closs.append(cls_loss.item())
        eloss.append(edge_loss.item() * 0.01)
        ploss.append(point_reg_loss.item() * 0.01)

        # clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # update weights
        optimizer.step()

        # calculate accuracy
        acc = accuracy(new_ellipse, gt_ellipse)
        train_acc.append(acc)

        # print status
        if i % print_freq == 0:
            print('Epoch: [{}]/[{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'cls_loss: {:.3f} - edge_loss: {:.3f} - point_reg_loss: {:.3f}\t'
                  'Accuracy: {:.3f}'.format(epoch, i, len(train_loader),
                                            sum(train_loss)/len(train_loss),
                                            sum(closs)/len(closs),
                                            sum(eloss)/len(eloss),
                                            sum(ploss)/len(ploss),
                                            sum(train_acc)/len(train_acc)))

def validate(val_loader, model, criterion):

    model.eval()

    val_closs = list()
    val_eloss = list()
    val_ploss = list()
    val_loss = list()
    val_acc = list()
    with torch.no_grad():
        for i, (img, img_edge, input_ellipse, label_ind, gt_ellipse) in enumerate(val_loader):

            # move to GPU, if available
            img = img.to(device)
            img_edge = img_edge.to(device)
            input_ellipse = input_ellipse.to(device)
            label_ind = label_ind.to(device)
            gt_ellipse = gt_ellipse.to(device)

            # forward prop.
            cls_feature, edge_feature, predict_offset = model(img, input_ellipse)
            label_ind = label_ind.view(-1)  # (1, bs)

            # update contour
            new_ellipse = input_ellipse + predict_offset
            new_ellipse[..., 0] = new_ellipse[..., 0] * 4.
            new_ellipse[..., 1] = new_ellipse[..., 1] * 4.

            # calculate loss
            cls_loss, edge_loss, point_reg_loss = criterion(cls_feature, label_ind, new_ellipse, gt_ellipse, edge_feature, img_edge)

            loss_value = cls_loss + 0.01 * edge_loss + point_reg_loss * 0.01

            # calculate accuracy
            acc = accuracy(new_ellipse, gt_ellipse)
            val_acc.append(acc)

            val_loss.append(loss_value.item())
            val_closs.append(cls_loss.item())
            val_eloss.append(edge_loss.item() * 0.01)
            val_ploss.append(point_reg_loss.item() * 0.01)

        val_accuracy = sum(val_acc) / len(val_acc)

        # print status
        print('Loss: {:.3f}\t'
              'cls_loss: {:.3f} - edge_loss: {:.3f} - point_reg_loss: {:.3f}\t'
              'Accuracy: {:.3f}'.format(sum(val_loss) / len(val_loss),
                                        sum(val_closs) / len(val_closs),
                                        sum(val_eloss) / len(val_eloss),
                                        sum(val_ploss) / len(val_ploss),
                                        val_accuracy))
        print('--------------------------------------------------------')
    return val_accuracy

if __name__ == '__main__':
    main()