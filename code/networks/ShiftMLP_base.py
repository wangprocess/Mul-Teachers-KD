import torch
import torch.nn.functional as F
from torch import nn


__all__ = ['ShiftMLP_b']

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size  # 5
        self.pad = shift_size // 2  # 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        # print('xn', xn.shape) # xn torch.Size([1, 160, 20, 20])
        xs = torch.chunk(xn, self.shift_size, 1)
        # print('xs', len(xs), xs[0].shape) # 5 torch.Size([1, 32, 20, 20])
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        # print('x_cat', x_cat.shape)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 160*1
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # print('x', x.shape)
        x = self.proj(x)
        # print('x-', x.shape)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # print('x-', x.shape)
        # print('H, W', H, W)
        return x, H, W


class ShiftMLP_b(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[16, 32, 64, 128, 256, 512, 1024], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1, 1, 1, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(64)
        self.ebn4 = nn.BatchNorm2d(128)
        self.ebn5 = nn.BatchNorm2d(256)
        self.ebn6 = nn.BatchNorm2d(512)

        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])
        self.norm6 = norm_layer(embed_dims[6])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 0,0,0

        self.blockt1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.blockt2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.blockt3 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.blockt4 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[4], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.blockt5 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[5], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.blockt6 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[6], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.patch_embed_t1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                embed_dim=embed_dims[1])

        self.patch_embed_t2 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                embed_dim=embed_dims[2])

        self.patch_embed_t3 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])

        self.patch_embed_t4 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])

        self.patch_embed_t5 = OverlapPatchEmbed(img_size=img_size // 64, patch_size=3, stride=2, in_chans=embed_dims[4],
                                                embed_dim=embed_dims[5])

        self.patch_embed_t6 = OverlapPatchEmbed(img_size=img_size // 128, patch_size=3, stride=2, in_chans=embed_dims[5],
                                                embed_dim=embed_dims[6])

        self.sum_conv = nn.Conv2d(2016, 512, kernel_size=1)

        # 分类
        # 定义全局平均池化层
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):

        B = x.shape[0]
        # Encoder
        # Conv Stage

        # Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # t1 torch.Size([1, 16, 128, 128])

        out_t1, H, W = self.patch_embed_t1(t1)  # ([1, 4096, 32]) 64 64
        for i, blk in enumerate(self.blockt1):
            out_t1 = blk(out_t1, H, W)
        out_t1 = self.norm1(out_t1)
        out_t1 = out_t1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t1', out_t1.shape)  # out_t1 torch.Size([1, 32, 64, 64])

        # Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # t2 torch.Size([1, 32, 64, 64])

        out_t2, H, W = self.patch_embed_t2(t2)
        for i, blk in enumerate(self.blockt2):
            out_t2 = blk(out_t2, H, W)
        out_t2 = self.norm2(out_t2)
        out_t2 = out_t2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t2', out_t2.shape)  # out_t2 torch.Size([1, 64, 32, 32])

        # Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # t3 torch.Size([1, 128, 32, 32])

        out_t3, H, W = self.patch_embed_t3(t3)
        for i, blk in enumerate(self.blockt3):
            out_t3 = blk(out_t3, H, W)
        out_t3 = self.norm3(out_t3)
        out_t3 = out_t3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t3', out_t3.shape)  # out_t3 torch.Size([1, 128, 16, 16])

        # Stage 4
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # t3 torch.Size([1, 256, 16, 16])

        out_t4, H, W = self.patch_embed_t4(t4)
        for i, blk in enumerate(self.blockt4):
            out_t4 = blk(out_t4, H, W)
        out_t4 = self.norm4(out_t4)
        out_t4 = out_t4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t4', out_t4.shape)  # out_t3 torch.Size([1, 256, 8, 8])

        # Stage 5
        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # t3 torch.Size([1, 512, 8, 8])

        out_t5, H, W = self.patch_embed_t5(t5)
        for i, blk in enumerate(self.blockt5):
            out_t5 = blk(out_t5, H, W)
        out_t5 = self.norm5(out_t5)
        out_t5 = out_t5.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t3', out_t3.shape)  # out_t3 torch.Size([1, 512, 4, 4])

        # Stage 6
        out = F.relu(F.max_pool2d(self.ebn6(self.encoder6(out)), 2, 2))
        t6 = out  # t3 torch.Size([1, 1024, 4, 4])

        out_t6, H, W = self.patch_embed_t6(t6)
        for i, blk in enumerate(self.blockt6):
            out_t6 = blk(out_t6, H, W)
        out_t6 = self.norm6(out_t6)
        out_t6 = out_t6.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print('out_t3', out_t3.shape)  # out_t3 torch.Size([1, 1024, 2, 2])

        # 把out_t1 [1, 32, 64, 64], out_t2 [1, 64, 32, 32], out_t3 [1, 128, 16, 16]合并
        out_t2 = F.interpolate(out_t2, size=(56, 56), mode='bilinear')
        out_t3 = F.interpolate(out_t3, size=(56, 56), mode='bilinear')
        out_t4 = F.interpolate(out_t4, size=(56, 56), mode='bilinear')
        out_t5 = F.interpolate(out_t5, size=(56, 56), mode='bilinear')
        out_t6 = F.interpolate(out_t6, size=(56, 56), mode='bilinear')
        out = F.interpolate(out, size=(56, 56), mode='bilinear')
        out_sum = torch.cat([out_t1, out_t2, out_t3, out_t4, out_t5, out_t6], 1)
        out_sum = self.sum_conv(out_sum)

        print('out_sum.shape', out_sum.shape)
        print('out.shape', out.shape)
        out = out + out_sum
        # 分类
        # 对卷积层的输出进行全局平均池化
        out = self.global_avgpool(out)
        out = out.view(x.size(0), -1)
        # print('pooled_out.shape', out.shape)

        return self.classifier(out)


# EOF
if __name__ == '__main__':
    x = torch.randn(5, 3, 224, 224)
    model = ShiftMLP_b(in_chans=3, num_classes=7, img_size=224)
    # 计算参数量
    params_num = sum(p.numel() for p in model.parameters())
    print("\nModle's Params: %.3fM" % (params_num / 1e6))
    y = model(x)
    print(y.size())
