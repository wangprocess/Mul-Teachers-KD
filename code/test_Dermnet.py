import argparse
import os

import numpy as np
import torch
import time

from tqdm import tqdm
import logging
import shutil
import sys

from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import transforms, meterics3
from networks.net_factory import net_factory
from dataloaders.datasets import DataSet_Dermnet_HQ_LQ_Test

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/DermnetData', help='the ROOT path/the path of dataset')
parser.add_argument('--exp', type=str,
                    default='DermnetData/MTKD', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='ShiftMLP_b', help='model_name')
parser.add_argument('--model_path', type=str,
                    default='../model/DermnetData/MTKD/XXXXXXX.pth', help='the test model path')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch size')

parser.add_argument('--device', type=str,
                    default='cuda', help='device name')
parser.add_argument('--data_quality', type=str,
                    default='LQ-Test', help='data quality name')
parser.add_argument('--data_num', type=int,
                    default=None, help='split the data')

args = parser.parse_args()


def test(args, snapshot_path):
    num_classes = args.num_classes
    # 网络的输出通道数 7
    device = args.device

    batch_size = args.batch_size

    writer = SummaryWriter(snapshot_path + '/log')

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    db_test = DataSet_Dermnet_HQ_LQ_Test(base_dir=args.root_path, split=args.data_quality, num=args.data_num, transform=transforms.Dermnet_Val_Transforms)
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1)

    model = create_model()
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    LOGITS = []
    PROBS = []
    LABELS = []

    with torch.no_grad():
        test_acc = 0
        test_auc = 0
        total_time = 0
        num_samples = 0
        for (img_test, label_test) in tqdm(test_loader):
            img_test, label_test = img_test.to(device), label_test.to(device)

            start_time = time.time()
            logits = model(img_test)
            end_time = time.time()

            total_time += (end_time - start_time)
            num_samples += batch_size

            probs_softmax = torch.softmax(logits, dim=1)
            LABELS.extend(np.asarray(label_test.to('cpu').numpy()))
            PROBS.extend(np.asarray(probs_softmax.to('cpu').numpy()))
            test_acc += meterics3.mean_accuracy(logits, label_test).cpu()

        AIT = total_time / num_samples
        test_auc = roc_auc_score(LABELS, PROBS, average='macro', multi_class='ovr')
        one_hot = torch.eye(5)

        AUROCs, Accus, Senss, Specs, pre, F1 = meterics3.compute_metrics_test(one_hot[LABELS], PROBS, competition=True)
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        pre_avg = np.array(pre).mean()
        F1_avg = np.array(F1).mean()
        print(AUROC_avg, Accus_avg, pre_avg, F1_avg)
        logging.info(
            'test_acc: %f, test_auc: %f, AIT in %s : %f s' %
            (test_acc / len(test_loader), test_auc, device, AIT))

        writer.add_scalar('info/_acc', test_acc)



if __name__ == '__main__':

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    test(args, snapshot_path)

