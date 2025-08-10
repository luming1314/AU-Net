from kornia.filters import SpatialGradient
from torch import Tensor

from modules.losses import *
import sys
import os
from utils.utils import RGB2YCrCb, YCbCr2RGB
from run import distributed_utils

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from modules.modules import SpatialTransformer, get_scheduler, gaussian_weights_init
from torch.cuda.amp import autocast, GradScaler

from modules.AU_Net import AU_Net


class AU_Net_Cls(nn.Module):
    def __init__(self, opts=None):
        super(AU_Net_Cls, self).__init__()
        self.iters_to_accumulate = 2
        self.opts = opts

        # parameters
        # lr = 0.001
        lr = 0.001
        self.lr = lr
        # lr *= opts.world_size  # The learning rate is multiplied by the number of parallel GPUs
        # encoders
        # self.DM = AU_Net()
        self.DM = AU_Net(first_conv_class=self.opts.first_conv_class)
        self.resume_flag = False
        self.ST = SpatialTransformer(256, 256, True)
        self.l1 = nn.L1Loss(reduction='none')
        self.spatial = SpatialGradient('diff')

        # optimizers
        self.DM_opt = torch.optim.Adam(
            self.DM.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        # self.FN_opt = torch.optim.Adam(
        #     self.FN.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.gradientloss = gradientloss()
        self.ncc_loss = ncc_loss()
        self.ssim_loss = ssimloss
        self.weights_sim = [1, 1, 0.2]
        self.weights_ssim1 = [0.3, 0.7]
        self.weights_ssim2 = [0.7, 0.3]

        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()

    def initialize(self):
        self.DM.apply(gaussian_weights_init)
        # self.FN.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.DM_sch = get_scheduler(self.DM_opt, opts, last_ep)
        # self.FN_sch = get_scheduler(self.FN_opt, opts, last_ep)
        return self.DM_sch

    def convert_bn_to_gn_single(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                num_features = child.num_features
                setattr(model, child_name, torch.nn.GroupNorm(num_groups=1, num_channels=num_features))
            else:
                self.convert_bn_to_gn_single(child)

    def convert_bn_to_gn(self):
        self.convert_bn_to_gn_single(self.DM)
        # self.convert_bn_to_gn_single(self.FN)

    def setgpu(self, device):
        self.DM.to(device)
        # self.FN.to(device)

    def setMultiGPU(self, gpu):
        # broadcast_buffers=False 使用ODConv需要写上
        self.gpu = gpu
        self.DM = torch.nn.parallel.DistributedDataParallel(self.DM, device_ids=[gpu], find_unused_parameters=False)
        # self.FN = torch.nn.parallel.DistributedDataParallel(self.FN, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    def setOptimizer(self):
        # optimizers
        self.DM_opt = torch.optim.Adam(
            self.DM.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.00001)
        # self.FN_opt = torch.optim.Adam(
        #     self.FN.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.00001)

    def setTrain(self):
        self.DM.train()
        # DenseFuse not train
        # self.FN.eval()
        # self.FN.train()
    def setZero_grad(self):
        self.DM_opt.zero_grad()
        # self.FN_opt.zero_grad()

    def test_forward(self, image_ir, image_vi):
        deformation = self.DM(image_ir, image_vi)
        image_ir_Reg = self.ST(image_ir, deformation['ir2vis'])
        image_fusion = self.FN(image_ir_Reg, image_vi)
        return image_fusion

    def generate_mask(self):
        flow = self.ST.grid + self.disp
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        if self.border_mask.device != goodmask.device:
            self.border_mask = self.border_mask.to(goodmask.device)
        self.goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        for i in range(2):
            self.goodmask = (self.AP(self.goodmask) > 0.3).float()

        flow = self.ST.grid - self.disp
        goodmask = F.grid_sample(self.goodmask, flow)
        self.goodmask_inverse = goodmask


    def forward(self, ir, vi):
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        ir_Y = ir[:, 0:1]
        fu = self.DM(ir_Y, vi_Y)['fus_ir2vis']
        fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)


        # disp = self.DM(ir, vi)['ir2vis']
        # ir_reg = self.ST(ir, disp)
        # vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        # fu = self.FN(ir_reg[:, 0:1], vi_Y)
        # fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
        return fu

    def registration_forward(self, ir, vi):
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        ir_Y = ir[:, 0:1]
        ir_reg = self.DM(ir_Y, vi_Y)['ir_warp']

        # disp = self.DM(ir, vi)['ir2vis']
        # ir_reg = self.ST(ir, disp)
        return ir_reg

    def fusion_forward(self, ir, vi):
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        fu = self.FN(ir[:, 0:1], vi_Y)
        fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
        return fu

    def train_forward_RF(self):
        b = self.image_ir_warp_RGB.shape[0]
        # todo now
        # input image
        self.image_vi_Y, self.image_vi_Cb, self.image_vi_Cr = RGB2YCrCb(self.image_vi_RGB)
        self.image_vi_warp_Y, self.image_vi_warp_Cb, self.image_vi_warp_Cr = RGB2YCrCb(self.image_vi_warp_RGB)
        self.image_ir_Y = self.image_ir_RGB[:, 0:1, ...]
        self.image_ir_warp_Y = self.image_ir_warp_RGB[:, 0:1, ...]

        self.image_vis_fake_Y, self.image_vis_fake_Cb, self.image_vis_fake_Cr = RGB2YCrCb(self.image_vis_fake_RGB)
        self.image_ir_fake_Y = self.image_ir_fake_RGB[:, 0:1, ...]

        ir_stack = torch.cat([self.image_ir_warp_Y, self.image_ir_Y])
        vi_stack = torch.cat([self.image_vi_Y, self.image_vi_warp_Y])

        deformation = self.DM(ir_stack, vi_stack, type='bi')
        self.down2 = deformation['down2']
        self.down4 = deformation['down4']
        self.down8 = deformation['down8']

        self.deformation_1['vis2ir'], self.deformation_2['vis2ir'] = deformation['vis2ir'][0:b, ...], deformation[
                                                                                                          'vis2ir'][b:,
                                                                                                      ...]
        self.deformation_1['ir2vis'], self.deformation_2['ir2vis'] = deformation['ir2vis'][0:b, ...], deformation[
                                                                                                          'ir2vis'][b:,
                                                                                                      ...]

        self.image_ir_Reg_Y, self.image_ir_warp_fake_Y = deformation['ir_warp'][0:b, ...], deformation['ir_warp'][b:, ...]
        self.image_vi_warp_fake_Y, self.image_vi_Reg_Y = deformation['vis_warp'][0:b, ...], deformation['vis_warp'][b:, ...]

        self.image_fusion_1, self.image_fusion_2 = deformation['fus_ir2vis'][0:b, ...], deformation['fus_vis2ir'][b:, ...],

        self.generate_mask()


    def update_RF(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp, image_vis_fake, image_ir_fake):
        self.image_ir_RGB = image_ir
        self.image_vi_RGB = image_vi
        self.image_ir_warp_RGB = image_ir_warp
        self.image_vi_warp_RGB = image_vi_warp
        self.disp = disp

        self.image_vis_fake_RGB = image_vis_fake
        self.image_ir_fake_RGB = image_ir_fake
        self.train_forward_RF()
        self.backward_RF()
        # nn.utils.clip_grad_norm_(self.DM.parameters(), 5)
        # nn.utils.clip_grad_norm_(self.FN.parameters(), 5)
        # self.DM_opt.step()
        # self.FN_opt.step()
        #
        # self.DM_opt.zero_grad()
        # self.FN_opt.zero_grad()
    def backward_RF_main(self, scaler: GradScaler, it):
        if self.opts.gn:
            self.loss_total = self.loss_total / self.iters_to_accumulate
        self.scaler = scaler
        self.scaler.scale(self.loss_total).backward()
        if self.opts.gn:
            if (it + 1) % self.iters_to_accumulate == 0:
                nn.utils.clip_grad_norm_(self.DM.parameters(), 5)
                # nn.utils.clip_grad_norm_(self.FN.parameters(), 5)
                self.scaler.step(self.DM_opt)
                # self.scaler.step(self.FN_opt)
                self.scaler.update()

                self.DM_opt.zero_grad()
                # self.FN_opt.zero_grad()
        else:
            nn.utils.clip_grad_norm_(self.DM.parameters(), 5)
            # nn.utils.clip_grad_norm_(self.FN.parameters(), 5)
            self.scaler.step(self.DM_opt)
            # self.scaler.step(self.FN_opt)
            self.scaler.update()

            self.DM_opt.zero_grad()
            # self.FN_opt.zero_grad()


    def imgloss(self, src, tgt, mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradientloss(src, tgt,
                                                                                                               mask)

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean()

    def fusloss(self, ir, vi, fu, weights=[1, 0, 0.5, 0]):
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
        grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
        loss_grad = 0.5 * F.l1_loss(grad_fus, grad_ir) + 0.5 * F.l1_loss(grad_fus, grad_vi)
        loss_ssim = 0.5 * self.ssim_loss(ir, fu) + 0.5 * self.ssim_loss(vi, fu)
        loss_intensity = 0.5 * F.l1_loss(fu, ir) + 0.5 * F.l1_loss(fu, vi)
        loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity
        return loss_intensity, loss_ssim, loss_grad, loss_total

    def fusloss_forRF(self, ir, vi, fu, weights=[0.6, 0.3, 0.1], mask=1):
        mask_ = (torch.logical_and(ir > 0, vi > 0) * mask).detach()
        if (fu > 2.0 / 255).sum() < 100:
            mask_ = 1
        ir = ir.detach()
        vi = vi.detach()
        fu = fu
        grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
        grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
        grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
        grad_joint = torch.max(grad_ir, grad_vi)
        loss_grad = (((grad_joint - grad_fus).abs().clamp(min=1e-9)) * mask_).mean()
        loss_ssim = (self.ssim_loss(ir, fu) + self.ssim_loss(vi, fu))
        # print(loss_ssim)
        intensity_joint = torch.max(vi, ir) * mask_
        Loss_intensity = F.l1_loss(fu * mask_, intensity_joint)
        return weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * Loss_intensity

    def Seg_loss(self, fused_image, label, seg_model):
        '''
        Using a pre-trained segmentation network, calculate the semantic loss between the fused segmentation result and the true label.
        :param fused_image:
        :param label:
        :param seg_model: The segmentation model is pre-loaded in the main function to avoid fully loading the segmentation model each time.
        :return seg_loss:
        The fused_image color space must be converted from YCbCr to RGB before being fed into the Seg_loss function.
        '''
        # Calculate semantic loss

        lb = torch.squeeze(label, 1)
        out, mid = seg_model(fused_image)
        out = F.softmax(out, 1)
        mid = F.softmax(mid, 1)
        seg_results = torch.argmax(out, dim=1, keepdim=True)
        lossp = lovasz_softmax(out, lb)
        loss2 = lovasz_softmax(mid, lb)
        seg_loss = lossp + 0.25 * loss2
        return seg_loss, seg_results

    def backward_RF_(self):
        loss_smooth_down2 = smoothloss(self.down2)
        loss_smooth_down4 = smoothloss(self.down4)
        loss_smooth_down8 = smoothloss(self.down8)
        loss_smooth = loss_smooth_down2 + loss_smooth_down4 + loss_smooth_down8

        loss_ncc = self.ncc_loss(self.image_vi_Y, self.image_ir_Reg_Y) + \
                   self.ncc_loss(self.image_vi_Reg_Y, self.image_ir_Y)
        loss_fuse = self.ssim_loss(self.image_fusion_1, self.image_vi_Y) + \
        self.ssim_loss(self.image_fusion_2, self.image_ir_Y) + \
        self.l1(self.image_fusion_1_grad, torch.zeros_like(self.image_fusion_1_grad)).mean() + \
        self.l1(self.image_fusion_2_grad, torch.zeros_like(self.image_fusion_2_grad)).mean()

        loss_ncc = loss_ncc.mean()
        loss_smooth = loss_smooth.mean()
        loss_fuse = loss_fuse.mean()

        loss_total = 100 * loss_ncc + 10 * loss_smooth + 200 * loss_fuse

        self.loss_reg_img = 0.
        self.loss_reg_field = 0.
        self.loss_fus = loss_fuse
        self.loss_smooth = loss_smooth
        self.loss_ncc = loss_ncc
        self.loss_total = loss_total

        pass




    def backward_RF(self):
        # Similarity loss for deformation
        # loss_reg_img = self.imgloss(self.image_ir_warp,self.image_ir_warp_fake)+self.imgloss(self.image_ir_Reg,self.image_ir)+\
        #     self.imgloss(self.image_vi_warp,self.image_vi_warp_fake)+self.imgloss(self.image_vi_Reg,self.image_vi)
        loss_reg_img = self.imgloss(self.image_ir_warp_Y, self.image_ir_warp_fake_Y, self.goodmask) + self.imgloss(
            self.image_ir_Reg_Y, self.image_ir_Y, self.goodmask * self.goodmask_inverse) + \
                       self.imgloss(self.image_vi_warp_Y, self.image_vi_warp_fake_Y, self.goodmask) + self.imgloss(
            self.image_vi_Reg_Y, self.image_vi_Y, self.goodmask * self.goodmask_inverse)
        loss_reg_field = self.weightfiledloss(self.image_ir_warp_Y, self.image_vi_warp_fake_Y,
                                              self.deformation_1['vis2ir'], self.disp.permute(0, 3, 1, 2)) + \
                         self.weightfiledloss(self.image_vi_warp_Y, self.image_ir_warp_fake_Y,
                                              self.deformation_2['ir2vis'], self.disp.permute(0, 3, 1, 2))
        # loss_smooth = smoothloss(self.deformation_1['vis2ir'])+smoothloss(self.deformation_1['ir2vis'])+\
        #     smoothloss(self.deformation_2['vis2ir'])+smoothloss(self.deformation_2['ir2vis'])
        loss_smooth_down2 = smoothloss(self.down2)
        loss_smooth_down4 = smoothloss(self.down4)
        loss_smooth_down8 = smoothloss(self.down8)
        loss_smooth = loss_smooth_down2 + loss_smooth_down4 + loss_smooth_down8
        loss_border_re = 0.1 * self.border_suppression(self.image_ir_Reg_Y,
                                                       self.goodmask_inverse) + 0.1 * self.border_suppression(
            self.image_vi_Reg_Y, self.goodmask_inverse) + \
                         self.border_suppression(self.image_ir_warp_fake_Y, self.goodmask) + self.border_suppression(
            self.image_vi_warp_fake_Y, self.goodmask)

        loss_style = self.ncc_loss(self.image_ir_Reg_Y * (self.goodmask * self.goodmask_inverse), self.image_ir_fake_Y  * (self.goodmask * self.goodmask_inverse)).mean() + \
                      self.ncc_loss(self.image_vi_Reg_Y * (self.goodmask * self.goodmask_inverse), self.image_vis_fake_Y * (self.goodmask * self.goodmask_inverse)).mean()

        loss_fus = self.fusloss_forRF(self.image_ir_Reg_Y, self.image_vi_Y, self.image_fusion_1,
                                      mask=self.goodmask * self.goodmask_inverse) + \
                   self.fusloss_forRF(self.image_ir_Y, self.image_vi_Reg_Y, self.image_fusion_2,
                                      mask=self.goodmask * self.goodmask_inverse)

        mask_ = torch.logical_and(self.image_ir_Y > 1e-5, self.image_vi_Y > 1e-5)
        mask_ = torch.logical_and(self.image_ir_Reg_Y > 1e-5, mask_)
        mask_ = torch.logical_and(self.image_vi_Reg_Y > 1e-5, mask_)
        mask_ = mask_ * self.goodmask * self.goodmask_inverse
        loss_ncc = self.imgloss(self.image_fusion_1, self.image_fusion_2, mask_)
        assert not loss_reg_img is None, 'loss_reg_img is None'
        assert not loss_reg_field is None, 'loss_reg_filed is None'
        assert not loss_smooth is None, 'loss_smooth is None'


        loss_total =  loss_reg_field + 1 * loss_smooth + 100 * loss_fus + loss_ncc + loss_border_re

        if self.opts.use_style:
            # 1.
            loss_total = loss_total + 10 * loss_style + loss_reg_img * 10



        loss_reg_img = distributed_utils.reduce_value(loss_reg_img, average=True)
        loss_reg_field = distributed_utils.reduce_value(loss_reg_field, average=True)
        loss_fus = distributed_utils.reduce_value(loss_fus, average=True)
        loss_smooth = distributed_utils.reduce_value(loss_smooth, average=True)
        loss_ncc = distributed_utils.reduce_value(loss_ncc, average=True)
        loss_total = distributed_utils.reduce_value(loss_total, average=True)
        loss_border_re = distributed_utils.reduce_value(loss_border_re, average=True)
        loss_style = distributed_utils.reduce_value(loss_style, average=True)

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field
        self.loss_fus = loss_fus
        self.loss_smooth = loss_smooth
        self.loss_ncc = loss_ncc
        self.loss_total = loss_total

        self.loss_border_re = loss_border_re
        self.loss_style = loss_style


    def update_lr(self):
        self.DM_sch.step()
        # self.FN_sch.step()
    def initialization(self, weights_path):
        state = {
            'DM': self.DM.state_dict(),
            # 'FN': self.FN.state_dict(),
            'DM_opt': self.DM_opt.state_dict(),
            # 'FN_opt': self.FN_opt.state_dict(),
            'ep': 0,
            'total_it': 0
        }
        torch.save(state, weights_path)


    def resume(self, model_dir, device, train=True):
        self.resume_flag = True
        checkpoint = torch.load(model_dir, map_location=device)
        # weight
        self.DM.load_state_dict({k: v for k, v in checkpoint['DM'].items() if k in self.DM.state_dict()}, strict=False)
        # self.FN.load_state_dict({k: v for k, v in checkpoint['FN'].items() if k in self.FN.state_dict()}, strict=False)
        # optimizer
        if train:
            # self.DM_opt.load_state_dict(checkpoint['DM_opt'])
            # self.FN_opt.load_state_dict(checkpoint['FN_opt'])
            # self.DM_opt.param_groups[0]['initial_lr'] = 0.001
            # self.FN_opt.param_groups[0]['initial_lr'] = 0.001

            self.DM_opt.param_groups[0]['initial_lr'] = self.lr
            # self.FN_opt.param_groups[0]['initial_lr'] = self.lr
        return checkpoint['ep'], checkpoint['total_it']
    def loadDenseFuse(self, model_dir, device):
        checkpoint = torch.load(model_dir, map_location=device)
        self.FN.load_state_dict(checkpoint)
        pass

    def save(self, filename, ep, total_it):
        state = {
            'DM': self.DM.module.state_dict(),
            # 'FN': self.FN.module.state_dict(),
            'DM_opt': self.DM_opt.state_dict(),
            # 'FN_opt': self.FN_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return
    def save_debug(self, filename, ep, total_it):
        state = {
            'DM': self.DM.state_dict(),
            # 'FN': self.FN.state_dict(),
            'DM_opt': self.DM_opt.state_dict(),
            # 'FN_opt': self.FN_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs1(self):
        images_ir = self.normalize_image(self.image_ir_RGB).detach()
        images_vi = self.normalize_image(self.image_vi_RGB).detach()
        images_fusion = self.normalize_image(self.image_fusion).detach()
        row = torch.cat((images_ir[0:1, ::], images_vi[0:1, ::], images_fusion[0:1, ::]), 3)
        return row

    def assemble_outputs(self):
        images_ir = self.normalize_image(self.image_ir_RGB).detach()
        images_vi = self.normalize_image(self.image_vi_RGB).detach()
        images_ir_warp = self.normalize_image(self.image_ir_warp_RGB).detach()
        images_vi_warp = self.normalize_image(self.image_vi_warp_RGB).detach()
        images_ir_Reg = self.normalize_image(self.image_ir_Reg_RGB).detach()
        images_vi_Reg = self.normalize_image(self.image_vi_Reg_RGB).detach()
        images_fusion_1 = self.normalize_image(self.image_fusion_1).detach()
        images_fusion_2 = self.normalize_image(self.image_fusion_2).detach()
        row1 = torch.cat(
            (images_ir[0:1, ::], images_ir_warp[0:1, ::], images_ir_Reg[0:1, ::], images_fusion_1[0:1, ::]), 3)
        row2 = torch.cat(
            (images_vi[0:1, ::], images_vi_warp[0:1, ::], images_vi_Reg[0:1, ::], images_fusion_2[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

        # self.image_display = torch.cat(
        #     (self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
        #      self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(),
        #      self.fake_A_recon[0:1].detach().cpu(), self.real_B_encoded[0:1].detach().cpu(),
        #      self.fake_A_encoded[0:1].detach().cpu(), self.fake_A_random[0:1].detach().cpu(),
        #      self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    def normalize_image(self, x):
        return x[:, 0:1, :, :]

    def gradient(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)

        # Binarization
        u = torch.where(u > torch.mean(u), 1., 0.)
        return u
