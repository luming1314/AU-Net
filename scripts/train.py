import os

import torch

from run.dataset import RegData
from run.model_AU_Net import AU_Net_Cls
from run.distributed_utils import init_distributed_mode, dist
from run.options import TrainOptions
from utils.logger import logger_config
from utils.saver import Saver
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
"""
author DDP
"""
def main_RF(opts):
    # Initialize each process environment
    writer = SummaryWriter(opts.tensorboardX_path, comment=opts.summary_name)
    if opts.debug:
        opts.rank = 0
    else:
        # use DDP
        init_distributed_mode(args=opts)
    rank = opts.rank
    device = torch.device(opts.device)
    batch_size = opts.batch_size
    weights_path = opts.resume



    if rank == 0:
        # Initialize the save environment
        sub_dir = 'mutil_128'
        log_dir = os.path.join(opts.result_dir, opts.name, sub_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'log.txt')
        if os.path.exists(log_path):
            os.remove(log_path)
        logger = logger_config(log_path=log_path, logging_name='Timer')

        # saver for display and output
        saver = Saver(opts)
        # daita loader
        logger.info('\n--- load dataset ---')
    dataset = RegData(opts)
    if opts.debug:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    else:
        # Assign training sample index to the process corresponding to each rank
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # Organize the sample index into a list of batch_size elements
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size=opts.batch_size, drop_last=True)
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        if rank == 0:
            print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   pin_memory=True,
                                                   num_workers=nw)
    if rank == 0:
        # model
        logger.info('\n--- load model is {%s} and use style_loss {%s}---' % (opts.first_conv_class, opts.use_style))
    model = AU_Net_Cls(opts)
    if opts.gn:
        model.convert_bn_to_gn()
    model.setgpu(device)
    # TODO Initialize weights Consider reading or automatically generating
    if os.path.exists(weights_path):
        model.resume(weights_path, device)
    else:
        # If there is no pre-trained weight, you need to save the weight in the first process and then load it in other processes to keep the initial weight consistent
        if rank == 0:
            model.initialization(weights_path)
        if not opts.debug:
            dist.barrier()
        # Note that you must specify the map_location parameter, otherwise the first GPU will occupy more resources.
        model.resume(weights_path, device)


    if not opts.debug:
        # DDP
        model.setMultiGPU(opts.gpu)

    # optimizer
    model.setOptimizer()
    scaler = GradScaler()

    ep0 = -1
    total_it = 0
    DM_sch = model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    if rank == 0:
        logger.info('start the training at epoch %d' % (ep0))
        min_score_total, min_score_reg = 10000000., 10000000.
    for ep in range(ep0, opts.n_ep):
        if rank == 0:
            Reg_Img_loss, Reg_Field_loss, Fusion_loss, NCC_loss, Total_loss, Smooth_loss, Border_re_loss, My_ncc_loss \
                , Grad_loss, Style_loss = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        if not opts.debug:
            train_sampler.set_epoch(ep)
        model.setTrain()
        model.setZero_grad()
        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation, image_vis_fake, image_ir_fake) in enumerate(train_loader):
                # input data
                image_ir = image_ir.to(device).detach()
                image_vi = image_vi.to(device).detach()
                image_ir_warp = image_ir_warp.to(device).detach()
                image_vi_warp = image_vi_warp.to(device).detach()
                deformation = deformation.to(device).detach()

                image_vis_fake = image_vis_fake.to(device).detach()
                image_ir_fake = image_ir_fake.to(device).detach()
                if len(image_ir.shape) > 4:
                    image_ir = image_ir.squeeze(1)
                    image_vi = image_vi.squeeze(1)
                    image_ir_warp = image_ir_warp.squeeze(1)
                    image_vi_warp = image_vi_warp.squeeze(1)
                    deformation = deformation.squeeze(1)

                    image_vis_fake = image_vis_fake.squeeze(1)
                    image_ir_fake = image_ir_fake.squeeze(1)
                # update model
                with autocast():
                    model.update_RF(image_ir, image_vi, image_ir_warp,
                                    image_vi_warp, deformation, image_vis_fake, image_ir_fake)
                model.backward_RF_main(scaler, it)
                if rank == 0:
                    Reg_Img_loss = Reg_Img_loss + model.loss_reg_img
                    Reg_Field_loss = Reg_Field_loss + model.loss_reg_field
                    Fusion_loss = Fusion_loss + model.loss_fus
                    NCC_loss = NCC_loss + model.loss_ncc
                    Total_loss = Total_loss + model.loss_total
                    Smooth_loss = Smooth_loss + model.loss_smooth

                    Border_re_loss = Border_re_loss + model.loss_border_re
                    Style_loss = Style_loss + model.loss_style



                torch.cuda.synchronize(device)
        # Update learning rate
        DM_sch.step()
        # FN_sch.step()
        # Print results
        if rank == 0:
            logger.info('(ep %d), lr %08f , Total Loss: %04f' % (ep, model.DM_opt.param_groups[0]['lr'], Total_loss / len(train_loader)))

            logger.info('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, NCC_loss: {:.4}, Fusion_loss: {:.4}, Smooth_loss: {:.4}'
                        ', Border_re_loss: {:.4}, Style_loss: {:.4}'.format(
                Reg_Img_loss / len(train_loader), Reg_Field_loss / len(train_loader), NCC_loss / len(train_loader),
                Fusion_loss / len(train_loader), Smooth_loss / len(train_loader), Border_re_loss / len(train_loader)
                , Style_loss / len(train_loader)))


            writer.add_scalar('Train/Total_loss', Total_loss / len(train_loader), ep)
            writer.add_scalar('Train/Reg_Field_loss', Reg_Field_loss / len(train_loader), ep)

            total_best, reg_best = False, False

            if (Total_loss / len(train_loader)) < min_score_total:
                min_score_total = (Total_loss / len(train_loader))
                logger.info('min_score_total: {:.4}'.format(min_score_total))
                total_best = True

            if (Reg_Field_loss / len(train_loader)) < min_score_reg:
                min_score_reg = (Reg_Field_loss / len(train_loader))
                logger.info('min_score_reg: {:.4}'.format(min_score_reg))
                reg_best = True

            saver.write_model(ep, opts.n_ep, model, debug=opts.debug, total_best=total_best, reg_best=reg_best, logger=logger)
    pass

if __name__ == '__main__':

    parser = TrainOptions()
    opts = parser.parse()
    main_RF(opts)