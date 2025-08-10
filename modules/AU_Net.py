from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layers import *
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
# from irnn import irnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class SpatialTransformer(nn.Module):
    def __init__(self, h, w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h, w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1] == 2:
            disp = disp.permute(0, 2, 3, 1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1], disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)


class DispEstimator(nn.Module):
    def __init__(self, channel, depth=4, norm=nn.BatchNorm2d, dilation=1):
        super(DispEstimator, self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel, channel, 3, act=None, norm=None, dilation=dilation, padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, act=None))
        # self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel + self.corrks ** 2
        dilation = 1
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256, 256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
        # self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        # self.AP3=nn.AvgPool2d(3,stride=1,padding=1)

    # def localcorr(self,feat1,feat2):
    #     feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
    #     feat1 = F.normalize(feat1,dim=1)
    #     feat2 = F.normalize(feat2,dim=1)
    #     b,c,h,w = feat2.shape
    #     feat2_smooth = KF.gaussian_blur2d(feat2,[9,9],[3,3])
    #     feat2_loc_blk = F.unfold(feat2_smooth,kernel_size=self.corrks,dilation=4,padding=4*(self.corrks-1)//2,stride=1).reshape(b,c,-1,h,w)
    #     localcorr = (feat1.unsqueeze(2)*feat2_loc_blk).sum(dim=1)
    #     localcorr = self.localcorrpropcessor(localcorr)
    #     corr = torch.cat([feat,localcorr],dim=1)
    #     return corr
    def localcorr(self, feat1, feat2):
        feat = self.featcompressor(torch.cat([feat1, feat2], dim=1))
        b, c, h, w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1, [13, 13], [3, 3], border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth, kernel_size=self.corrks, dilation=4, padding=2 * (self.corrks - 1),
                                 stride=1).reshape(b, c, -1, h, w)
        localcorr = (feat2.unsqueeze(2) - feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat, localcorr], dim=1)
        return corr

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0, 1, 0, 0] != w - 1 or self.scale[0, 0, 0, 0] != h - 1:
            self.scale = torch.FloatTensor([w, h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1, feat2)
        for i, layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr, [13, 13], [3, 3], border_type='replicate')
        disp = corr.clamp(min=-300, max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp / self.scale


class DispRefiner(nn.Module):
    def __init__(self, channel, dilation=1, depth=4):
        super(DispRefiner, self).__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None))
        oc = channel
        ic = channel + 2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(
                Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)

    def forward(self, feat1, feat2, disp):
        b = feat1.shape[0]
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))
        corr = torch.cat([feat, disp], dim=1)
        delta_disp = self.estimator(corr)
        disp = disp + delta_disp
        return disp


class Feature_extractor_unshare(nn.Module):
    def __init__(self, depth, base_ic, base_oc, base_dilation, norm):
        super(Feature_extractor_unshare, self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i % 2 == 1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(
                    ResConv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            else:
                feature_extractor.append(
                    Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=norm))
            ic = oc
            if i % 2 == 1 and i < depth - 1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)


class AU_Net(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2, first_conv_class='ODConv'):
        super(AU_Net, self).__init__()
        self.first_conv_class = first_conv_class
        self.num_pyramids = num_pyramids
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth, base_ic=1, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        # self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 2, base_oc * 2, kernel_size=3, stride=2, padding=1, dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(base_oc * 2, base_oc * 4, kernel_size=3, stride=1, padding=2, dilation=2, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 4, base_oc * 4, kernel_size=3, stride=2, padding=2, dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(base_oc * 4, base_oc * 8, kernel_size=3, stride=1, padding=4, dilation=4, norm=nn.InstanceNorm2d),
            Conv2d(base_oc * 8, base_oc * 8, kernel_size=3, stride=2, padding=4, dilation=4, norm=nn.InstanceNorm2d))
        self.matcher1 = DispEstimator(base_oc * 4, matcher_depth, dilation=4)
        self.matcher2 = DispEstimator(base_oc * 8, matcher_depth, dilation=2)
        self.refiner = DispRefiner(base_oc * 2, 1)
        self.grid_down = KU.create_meshgrid(64, 64).cuda()
        self.grid_full = KU.create_meshgrid(128, 128).cuda()
        self.scale = torch.FloatTensor([128, 128]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1


        if first_conv_class == 'ODConv':

            self.up_fus_3 = nn.Sequential(
                ConvTransposeBN2d(base_oc * 16, base_oc * 8, kernel_size=3, stride=2, padding=1, dilation=1),
                ODConvBN2d(base_oc * 8, base_oc * 8, kernel_size=3, stride=1, padding=1, dilation=1)
            )
            self.up_fus_2 = nn.Sequential(
                ConvTransposeBN2d(base_oc * 8, base_oc * 4, kernel_size=3, stride=2, padding=1, dilation=1),
                ODConvBN2d(base_oc * 4, base_oc * 4, kernel_size=3, stride=1, padding=1, dilation=1)
            )
            self.up_fus_1 = nn.Sequential(
                ConvTransposeBN2d(base_oc * 4, base_oc * 2, kernel_size=3, stride=2, padding=1, dilation=1),
                ODConvBN2d(base_oc * 2, base_oc * 2, kernel_size=3, stride=1, padding=1, dilation=1)
            )




        self.ST = SpatialTransformer(256, 256, True)

        self.decode1 = ConvBnTanh2d(base_oc * 2, 1)
        if first_conv_class == 'ODConv':
            self.rgdb1 = RGBD(32, 48)
            self.decode2 = ConvBnLeakyRelu2d(48, 32)
        else:
            self.decode2 = ConvBnLeakyRelu2d(32, 32)



    def match(self, feat11, feat12, feat21, feat22, feat31, feat32):
        # compute scale (w,h)
        if self.scale[0, 1, 0, 0] * 2 != feat11.shape[2] - 1 or self.scale[0, 0, 0, 0] * 2 != feat11.shape[3] - 1:
            self.h, self.w = feat11.shape[2], feat11.shape[3]
            self.scale = torch.FloatTensor([self.w, self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat11.device)

        # estimate disp src(feat1) to tgt(feat2) in low resolution
        disp2_raw = self.matcher2(feat31, feat32)

        # todo warp feat31
        disp3_warp = disp2_raw
        if disp3_warp.shape[2] != self.grid_down.shape[1] or disp3_warp.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat31.shape[2], feat31.shape[3]).cuda()
        feat31_warp = F.grid_sample(feat31, self.grid_down.to(feat31.device) + disp3_warp.permute(0, 2, 3, 1).to(feat31.device))


        # upsample disp and grid
        disp2 = F.interpolate(disp2_raw, [feat21.shape[2], feat21.shape[3]], mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2], feat21.shape[3]).cuda()

        # warp the last src(fea1) to tgt(feat2) with disp2
        feat21 = F.grid_sample(feat21, self.grid_down.to(feat21.device) + disp2.permute(0, 2, 3, 1).to(feat21.device) )

        # estimate disp src(feat1) to tgt(feat2) in low resolution
        disp1_raw = self.matcher1(feat21, feat22)

        # todo warp feat21
        disp2_warp = disp1_raw
        if disp2_warp.shape[2] != self.grid_down.shape[1] or disp2_warp.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2], feat21.shape[3]).cuda()
        feat21_warp = F.grid_sample(feat21, self.grid_down.to(feat21.device) + disp2_warp.permute(0, 2, 3, 1).to(feat21.device))


        # upsample
        disp1 = F.interpolate(disp1_raw, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        disp2 = F.interpolate(disp2, [feat11.shape[2], feat11.shape[3]], mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2], feat11.shape[3]).cuda()

        # warp
        feat11 = F.grid_sample(feat11, self.grid_full.to(feat11.device) + (disp1 + disp2).permute(0, 2, 3, 1).to(feat11.device))

        # finetune
        disp_scaleup = (disp1 + disp2) * self.scale
        disp = self.refiner(feat11, feat12, disp_scaleup)
        disp = KF.gaussian_blur2d(disp, [17, 17], [5, 5], border_type='replicate') / self.scale

        # todo warp feat11
        disp1_warp = disp
        if disp1_warp.shape[2] != self.grid_full.shape[1] or disp1_warp.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2], feat11.shape[3]).cuda()
        feat11_warp = F.grid_sample(feat11, self.grid_full.to(feat11.device) + disp1_warp.permute(0, 2, 3, 1).to(feat11.device))

        if self.training:
            return disp, disp_scaleup / self.scale, disp2, {'feat3_warp': feat31_warp, 'feat2_warp': feat21_warp, 'feat1_warp': feat11_warp}
        return disp,  {'feat3_warp': feat31_warp, 'feat2_warp': feat21_warp, 'feat1_warp': feat11_warp}, None, None

    def fusion(self, feat_warp: List, feat_gt: List):
        """

        :param feat_warp: ir_list
        :param feat_gt: vi_list
        :return:
        """
        b, c, h, w = feat_warp[3].size()

        # feat_warp[0], feat_gt[0] = self.ir_gdb_3(feat_warp[0]), self.vi_gdb_3(feat_gt[0])
        # feat_warp[1], feat_gt[1] = self.ir_gdb_2(feat_warp[1]), self.vi_gdb_2(feat_gt[1])
        # feat_warp[2], feat_gt[2] = self.ir_gdb_1(feat_warp[2]), self.vi_gdb_1(feat_gt[2])
        # feat_warp[3], feat_gt[3] = self.ir_gdb_0(feat_warp[3]), self.vi_gdb_0(feat_gt[3])

        feat3_concat = torch.cat([feat_warp[0], feat_gt[0]], dim=1)
        feat2_concat = torch.cat([feat_warp[1], feat_gt[1]], dim=1)
        feat1_concat = torch.cat([feat_warp[2], feat_gt[2]], dim=1)
        feat0_concat = torch.cat([feat_warp[3], feat_gt[3]], dim=1)

        fus_feat = feat0_concat

        if self.first_conv_class == 'ODConv':

            x = F.interpolate(self.up_fus_3(feat3_concat), [feat2_concat.shape[2], feat2_concat.shape[3]], mode='bilinear')
            # feat2_concat = feat2_concat + x
            feat2_concat = torch.max(feat2_concat, x)
            # feat2_concat = torch.cat([feat2_concat, x], dim=1)
            # feat2_concat = self.down_3(feat2_concat)
            # feat2_concat = self.rgdb3(feat2_concat)

            x = F.interpolate(self.up_fus_2(feat2_concat), [feat1_concat.shape[2], feat1_concat.shape[3]], mode='bilinear')
            # feat1_concat = feat1_concat + x
            feat1_concat = torch.max(feat1_concat, x)

            # feat1_concat = torch.cat([feat1_concat, x], dim=1)
            # feat1_concat = self.down_2(feat1_concat)
            # feat1_concat = self.rgdb2(feat1_concat)

            x = F.interpolate(self.up_fus_1(feat1_concat), [h, w], mode='bilinear')
            # fus_feat = feat0_concat + x
            fus_feat = torch.max(feat0_concat, x)


            # feat0_concat = torch.cat([feat0_concat, x], dim=1)
            # feat0_concat = self.down_1(feat0_concat)
            fus_feat = self.rgdb1(fus_feat)

        fus_feat = self.decode2(fus_feat)

        fus = self.decode1(fus_feat)

        return fus

    def forward(self, src, tgt, type='ir2vis'):
        # TODO Shared Feature Extraction module
        b, c, h, w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)
        feat0 = torch.cat([feat01, feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11, feat12 = feat1[0:b], feat1[b:]
        feat21, feat22 = feat2[0:b], feat2[b:]
        feat31, feat32 = feat3[0:b], feat3[b:]
        disp_12 = None
        disp_21 = None
        fus_1 = None
        fus_2 = None
        src_warp = None
        tgt_warp = None

        # 'bi' is bidirectional registration and fusion, only used in the training phase
        # 'ir2vis' is a one-way registration and fusion, ir is a moving image, vi is a fixed image, and is only used in the testing phase
        # 'vis2ir' is a one-way registration and fusion, vi is a moving image, ir is a fixed image, and is only used in the testing phase

        if type == 'bi':
            # TODO  Registration module
            disp_12, disp_12_down4, disp_12_down8, feat_warp_1 = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_21, disp_21_down4, disp_21_down8, feat_warp_2 = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            t = torch.cat([disp_12, disp_21, disp_12_down4, disp_21_down4, disp_12_down8, disp_21_down8])
            t = F.interpolate(t, [h, w], mode='bilinear')
            down2, down4, donw8 = torch.split(t, 2 * b, dim=0)
            disp_12_, disp_21_ = torch.split(down2, b, dim=0)


            feat31_warp, feat21_warp, feat11_warp = feat_warp_1['feat3_warp'], feat_warp_1['feat2_warp'], feat_warp_1['feat1_warp']
            feat32_warp, feat22_warp, feat12_warp = feat_warp_2['feat3_warp'], feat_warp_2['feat2_warp'], feat_warp_2['feat1_warp']

            img_stack = torch.cat([src, tgt])
            disp_stack = torch.cat([disp_12_, disp_21_])
            img_warp_stack = self.ST(img_stack, disp_stack)
            src_warp, tgt_warp = torch.split(img_warp_stack, b, dim=0)
            # src_warp_feat, tgt_warp_feat = self.feature_extractor_unshare1(src_warp), self.feature_extractor_unshare2(tgt_warp)


            self.grid_down = KU.create_meshgrid(feat01.shape[2], feat01.shape[3]).cuda()
            feat01_warp = F.grid_sample(feat01, self.grid_down + disp_12_.permute(0, 2, 3, 1))
            feat02_warp = F.grid_sample(feat02, self.grid_down + disp_21_.permute(0, 2, 3, 1))

            # TODO  Fusion module
            fus_1 = self.fusion([feat31_warp, feat21_warp, feat11_warp, feat01_warp], [feat32, feat22, feat12, feat02])
            fus_2 = self.fusion([feat31, feat21, feat11, feat01], [feat32_warp, feat22_warp, feat12_warp, feat02_warp])
            pass
        elif type == 'ir2vis':
            # TODO  Registration module
            disp_12, feat_warp_1, _, _ = self.match(feat11, feat12, feat21, feat22, feat31, feat32)
            disp_12 = F.interpolate(disp_12, [h, w], mode='bilinear')


            feat31_warp, feat21_warp, feat11_warp = feat_warp_1['feat3_warp'], feat_warp_1['feat2_warp'], feat_warp_1['feat1_warp']
            src_warp = self.ST(src, disp_12)
            # src_warp_feat = self.feature_extractor_unshare1(src_warp)


            self.grid_down = KU.create_meshgrid(feat01.shape[2], feat01.shape[3]).cuda()
            feat01_warp = F.grid_sample(feat01, self.grid_down.to(feat01.device) + disp_12.permute(0, 2, 3, 1))

            # TODO  Fusion module
            fus_1 = self.fusion([feat31_warp, feat21_warp, feat11_warp, feat01_warp], [feat32, feat22, feat12, feat02])
        elif type == 'vis2ir':
            # TODO  Registration module
            disp_21, feat_warp_2, _, _ = self.match(feat12, feat11, feat22, feat21, feat32, feat31)
            disp_21 = F.interpolate(disp_21, [h, w], mode='bilinear')


            feat32_warp, feat22_warp, feat12_warp = feat_warp_2['feat3_warp'], feat_warp_2['feat2_warp'], feat_warp_2['feat1_warp']
            tgt_warp = self.ST(tgt, disp_21)
            # tgt_warp_feat = self.feature_extractor_unshare2(tgt_warp)


            self.grid_down = KU.create_meshgrid(feat01.shape[2], feat01.shape[3]).cuda()
            feat02_warp = F.grid_sample(feat02, self.grid_down + disp_21.permute(0, 2, 3, 1))

            # TODO  Fusion module
            fus_2 = self.fusion([feat31, feat21, feat11, feat01], [feat32_warp, feat22_warp, feat12_warp, feat02_warp])


        if self.training:
            return {'ir2vis': disp_12_, 'vis2ir': disp_21_, 'fus_ir2vis': fus_1, 'fus_vis2ir': fus_2, 'ir_warp': src_warp, 'vis_warp': tgt_warp,
                    'down2': down2,
                    'down4': down4,
                    'down8': donw8}
        return {'ir2vis': disp_12, 'vis2ir': disp_21, 'fus_ir2vis': fus_1, 'fus_vis2ir': fus_2, 'ir_warp': src_warp, 'vis_warp': tgt_warp}


def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                   float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass


if __name__ == '__main__':
    # matcher = AU_Net().cuda()
    matcher = AU_Net(first_conv_class='static').cuda()
    ir = torch.rand(2, 1, 512, 512).cuda()
    vis = torch.rand(2, 1, 512, 512).cuda()
    disp = matcher(ir, vis, 'bi')
    print('ok')
