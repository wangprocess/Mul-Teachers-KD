import argparse
import logging
import os
import random
import shutil
import sys
import math
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from dataloaders import utils
from dataloaders.datasets import DataSet_ISIC_Teacher_or_Stu
from networks.net_factory import net_factory
from utils import losses, transforms, meterics, plots

# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

# Transforms.py


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Dermopathy', help='the ROOT path/the path of dataset')
parser.add_argument('--exp', type=str,
                    default='Dermopathy/teacher_alone', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='UASwinTv2b', help='model_name')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[3, 256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=666, help='random seed')
parser.add_argument('--num_classes', type=int, default=7,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')

# pretrain
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    # 学习率 0.01
    num_classes = args.num_classes
    # 网络的输出通道数 7
    batch_size = args.batch_size
    # batch大小 4
    max_epochs = args.max_epochs
    # 最大的epoch数目
    pretrain_model = args.pretrain_model

    # None

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # using pretrain?
    if pretrain_model:
        model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = DataSet_ISIC_Teacher_or_Stu(base_dir=args.root_path, split="train", num=None, transform=transforms.Train_Transforms)
    db_val = DataSet_ISIC_Teacher_or_Stu(base_dir=args.root_path, split="val", transform=transforms.Val_Transforms)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss().cuda()
    focal_loss = losses.FocalLoss().cuda()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    max_iterations = max_epochs * len(trainloader)
    iter_num = 0

    logging.info("The Number of epoch : {} ".format(max_epochs))

    iterator = tqdm(range(1, max_epochs+1), ncols=70)
    # 生成进度条
    train_loss_to_plot = []
    train_acc_to_plot = []
    val_loss_to_plot = []
    val_acc_to_plot = []
    best_performance = 0.0
    for epoch_num in iterator:
        # ----------------------------------------------------------------
        # Train phase
        batch_loss, batch_acc = 0, 0
        batch_loss_ce, batch_loss_focal = 0, 0
        epoch_loss, epoch_acc = 0, 0
        model.train()
        for i_batch, (img, label) in enumerate(trainloader):
            img, label = img.cuda(), label.cuda()
            outputs = model(img)
            print(outputs.shape)
            print(label.shape)
            loss_ce = ce_loss(outputs, label.long())

            # focal loss
            loss_focal = focal_loss(outputs, label.long())
            loss = 0.3 * loss_ce + 0.7 * loss_focal
            batch_loss += loss.detach().item()
            batch_loss_ce += loss_ce.detach().item()
            batch_loss_focal += loss_focal.detach()
            batch_acc += meterics.mean_accuracy(outputs, label).cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 动态调整学习率
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            if (i_batch+1) % 100 == 0:
                logging.info(
                    'epoch: %d iteration %d / %d: mean_loss_per_100iter: %f, loss_ce: %f, loss_focal: %f || mean_acc_per_100iter: %f' %
                    (epoch_num, i_batch+1, len(trainloader), batch_loss/(i_batch+1), batch_loss_ce/(i_batch+1), batch_loss_focal/(i_batch+1), batch_acc/(i_batch+1)))
                print('-' * 50)

        epoch_loss = batch_loss/len(trainloader)
        epoch_acc = batch_acc/len(trainloader)
        train_loss_to_plot.append(epoch_loss)
        train_acc_to_plot.append(epoch_acc)

        # --------------------------------------------------------------------
        # Validation phase
        val_batch_loss, val_batch_acc = 0, 0
        val_batch_loss_ce, val_batch_loss_focal = 0, 0
        val_epoch_loss, val_epoch_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for i_batch_val, (img_val, label_val) in enumerate(valloader):
                img_val, label_val = img_val.cuda(), label_val.cuda()
                preds = model(img_val)

                loss_ce = ce_loss(preds, label_val.long())
                loss_focal = focal_loss(preds, label_val.long())
                loss = 0.3 * loss_ce + 0.7 * loss_focal
                val_batch_loss += loss.detach().item()
                val_batch_loss_ce += loss_ce.detach().item()
                val_batch_loss_focal += loss_focal.detach()
                val_batch_acc += meterics.mean_accuracy(preds, label_val).cpu()

            logging.info(
                'epoch: %d | mean_val_loss: %f, val_loss_ce: %f, val_loss_focal: %f || mean_val_acc: %f' %
                (epoch_num, val_batch_loss / len(valloader), val_batch_loss_ce / len(valloader),
                 val_batch_loss_focal / len(valloader), val_batch_acc / len(valloader)))
            val_epoch_loss = val_batch_loss / len(valloader)
            val_epoch_acc = val_batch_acc / len(valloader)
            val_loss_to_plot.append(val_epoch_loss)
            val_acc_to_plot.append(val_epoch_acc)
            writer.add_scalar('info/val_mean_acc', val_epoch_acc, iter_num)
    #
            if val_epoch_acc > best_performance:
                best_performance = val_epoch_acc
                save_mode_path = os.path.join(snapshot_path,
                                              'iter_{}_acc_{:.4f}.pth'.format(
                                                  iter_num, best_performance))
                save_best = os.path.join(snapshot_path,
                                         '{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)

            logging.info(
                'iteration %d : acc : %f ' % (iter_num, val_epoch_acc))

    writer.close()
    plots.plot_loss_and_acc(train_loss_to_plot, val_loss_to_plot, train_acc_to_plot, val_acc_to_plot, "UASwinV2B")
    return "Training Finished!"


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)