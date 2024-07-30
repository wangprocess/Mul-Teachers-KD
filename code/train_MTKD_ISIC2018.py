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
from dataloaders.datasets import DataSet_ISIC_Teacher_or_Stu, DataSet_ISIC_MT_SSL, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, transforms, meterics, plots, ramps

# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

# Transforms.py


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Dermopathy', help='the ROOT path/the path of dataset')
parser.add_argument('--exp', type=str,
                    default='Dermopathy/MTKD', help='experiment_name')
parser.add_argument('--t_model', type=str,
                    default='UASwinTv2b', help='teacher_model_name')
parser.add_argument('--t_model_path', type=str,
                    default='XXXXXXXXXX.pth', help='teacher_model_path')
parser.add_argument('--s_model', type=str,
                    default='ShiftMLP_s', help='student_model_name')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[3, 224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=666, help='random seed')
parser.add_argument('--num_classes', type=int, default=7,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=16,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4006,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.90, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
# parser.add_argument('--consistency', type=float,
#                     default=100, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=50.0, help='consistency_rampup')

# KD
parser.add_argument('--T', type=float, default='2',
                    help='Temperature of distillation')
parser.add_argument('--alpha', type=float, default='0.7',
                    help='alpha')

# pretrain
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

args = parser.parse_args()


def get_labeled_number(dataset, ori_img_num):
    ref_dict = None
    if "Dermopathy" in dataset:  # 1-19978 are HQ
        ref_dict = {"4006": 19978}
    else:
        print("Error")
    return ref_dict[str(ori_img_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.alpha * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


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

    def create_model(net_type, ema=False):
        model = net_factory(net_type=net_type, in_chns=3,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    student_model = create_model(net_type=args.s_model)
    teacher_model = create_model(net_type=args.t_model)
    # read weights of teacher model
    teacher_model.load_state_dict(torch.load(args.t_model_path))
    teacher_model.eval()
    # using pretrain?
    if pretrain_model:
        student_model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")

    ema_model = create_model(net_type=args.s_model, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = DataSet_ISIC_MT_SSL(base_dir=args.root_path, split="train", num=None,
                                   transform=transforms.Train_Transforms)
    db_val = DataSet_ISIC_MT_SSL(base_dir=args.root_path, split="val", transform=transforms.Val_Transforms)

    total_sample = len(db_train)
    labeled_sample = get_labeled_number(args.root_path, args.labeled_num)
    print("Total samples is: {}, labeled samples is: {}".format(
        total_sample, labeled_sample
    ))
    labeled_idx = list(range(0, labeled_sample))
    unlabeled_idx = list(range(labeled_sample, total_sample))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idx, unlabeled_idx, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss().cuda()
    focal_loss = losses.FocalLoss().cuda()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    max_iterations = max_epochs * len(trainloader)
    iter_num = 0

    logging.info("The Number of epoch : {} ".format(max_epochs))

    iterator = tqdm(range(1, max_epochs + 1), ncols=70)
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
        batch_KD_loss = 0
        batch_supervised_loss, batch_consistency_loss = 0, 0
        epoch_loss, epoch_acc = 0, 0
        student_model.train()
        for i_batch, (img, label) in enumerate(trainloader):
            all_img, all_label = img.cuda(), label.cuda()
            unlabeled_img = all_img[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_img) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_img + noise

            outputs = student_model(all_img)
            teacher_outputs = teacher_model(all_img)

            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs], all_label[:args.labeled_bs].long())

            # focal loss
            loss_focal = focal_loss(outputs[:args.labeled_bs], all_label[:args.labeled_bs].long())
            supervised_loss = 0.3 * loss_ce + 0.7 * loss_focal

            KD_loss = losses.KD_loss(outputs[:args.labeled_bs], teacher_outputs[:args.labeled_bs], args.T)

            consistency_weight = get_current_consistency_weight(epoch_num)

            if epoch_num < 10:
                consistency_loss = 0.0
                consistency_weight = 0.0
            elif epoch_num > 60:
                consistency_weight = 100
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled.bs:] - ema_output_soft) ** 2
                )

            loss = supervised_loss * (1 - args.alpha) + consistency_weight * consistency_loss + KD_loss * (args.alpha - consistency_weight)

            batch_supervised_loss += supervised_loss.detach().item()
            batch_loss_ce += loss_ce.detach().item()
            batch_loss_focal += loss_focal.detach()
            batch_consistency_loss += consistency_weight * consistency_loss
            batch_KD_loss += KD_loss.detach()
            batch_loss += loss.detach().item()
            batch_acc += meterics.mean_accuracy(outputs[:args.labeled_bs], all_label[:args.labeled_bs]).cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(student_model, ema_model, args.ema_decay, iter_num)
            # 动态调整学习率
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            if (i_batch + 1) % 100 == 0:
                logging.info(
                    'epoch: %d iteration %d / %d: mean_loss_per_100iter: %f, loss_supervised: %f, loss_KD: %f, loss_ce: %f, loss_focal: %f, loss_ssl: %f|| mean_acc_per_100iter: %f' %
                    (
                        epoch_num, i_batch + 1, len(trainloader), batch_loss / (i_batch + 1),
                        batch_supervised_loss / (i_batch + 1), batch_KD_loss / (i_batch + 1),
                        batch_loss_ce / (i_batch + 1), batch_loss_focal / (i_batch + 1),
                        batch_consistency_loss / (i_batch + 1), batch_acc / (i_batch + 1)
                    )
                )
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', batch_loss / (i_batch + 1), iter_num)
                writer.add_scalar('info/loss_ce', batch_loss_ce / (i_batch + 1), iter_num)
                writer.add_scalar('info/loss_focal', batch_loss_focal / (i_batch + 1), iter_num)
                writer.add_scalar('info/consistency_loss',
                                  consistency_loss, iter_num)
                writer.add_scalar('info/consistency_weight',
                                  consistency_weight, iter_num)
                print('-' * 50)

        epoch_loss = batch_loss / len(trainloader)
        epoch_acc = batch_acc / len(trainloader)
        train_loss_to_plot.append(epoch_loss)
        train_acc_to_plot.append(epoch_acc)

        # --------------------------------------------------------------------
        # Validation phase
        val_batch_loss, val_batch_acc = 0, 0
        val_batch_loss_ce, val_batch_loss_focal = 0, 0
        val_epoch_loss, val_epoch_acc = 0, 0
        student_model.eval()
        with torch.no_grad():
            for i_batch_val, (img_val, label_val) in enumerate(valloader):
                img_val, label_val = img_val.cuda(), label_val.cuda()
                preds = student_model(img_val)

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
                                              'iter_{}_acc_{}.pth'.format(
                                                  iter_num, round(best_performance, 4)))
                save_best = os.path.join(snapshot_path,
                                         '{}_best_model.pth'.format(args.model))
                torch.save(student_model.state_dict(), save_mode_path)
                torch.save(student_model.state_dict(), save_best)

            logging.info(
                'iteration %d : acc : %f ' % (iter_num, val_epoch_acc))
            student_model.train()

    writer.close()
    plots.plot_loss_and_acc(train_loss_to_plot, val_loss_to_plot, train_acc_to_plot, val_acc_to_plot, "MTKD-ISIC2018")
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

    snapshot_path = "../model/{}/{}".format(args.exp, args.s_model)
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
