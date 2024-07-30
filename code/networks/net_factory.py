from .ShiftMLP_small import ShiftMLP_s
from .ShiftMLP_base import ShiftMLP_b
from .UASwinTv2b import UASwinV2B
from .UASwinTv2s import UASwinV2S
from .UASwinTv2t import UASwinV2T

import torchvision.models as models
import torch.nn as nn

def net_factory(net_type="ShiftMLP_s", in_chns=3, class_num=7, image_size=224):
    if net_type == "ShiftMLP_s":
        net = ShiftMLP_s(in_chns=in_chns, num_classes=class_num)
    elif net_type == "ShiftMLP_b":
        net = ShiftMLP_b(in_chns=in_chns, num_classes=class_num)
    elif net_type == "UASwinTv2b":
        net = UASwinV2B(in_chns=in_chns, num_classes=class_num, image_size=image_size)
    elif net_type == "UASwinTv2s":
        net = UASwinV2S(in_chns=in_chns, num_classes=class_num, image_size=image_size)
    elif net_type == "UASwinTv2t":
        net = UASwinV2T(in_chns=in_chns, num_classes=class_num, image_size=image_size)
    else:
        net = None
    return net.cuda()
