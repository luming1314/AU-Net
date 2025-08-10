import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='./datasets/train/your_data_name', help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    
    # ouptput related
    self.parser.add_argument('--name', type=str, default='FS_MSRS', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=50, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=50, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=2000, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=1600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default='./checkpoint/MSRS.pth', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--stage', type=str, default='FS', help='reg&fus (RF) or fus&seg (FS)')
    
    #segmentation related
    self.parser.add_argument('--dataroot_val', type=str, default='./dataset/test/MSRS/', help="data for segmentation validation")

    # We add parameters
    self.parser.add_argument('--first_conv_class', type=str, default='static',
                             help="please choice the first conv class from ['static', 'deformable', 'dynamic', 'd2Conv', '']")
    self.parser.add_argument('--use_style', action='store_true', help='Choose whether to use style transfer loss')

    #tensorboardX
    self.parser.add_argument('--summary_name', type=str, default='Diff_RF',
                        help='Name of the tensorboard summmary')
    self.parser.add_argument('--tensorboardX_path', type=str, default='logs/tensorboardX',
                        help='Path of the tensorboard summmary')

    # train
    # Do not change this parameter, the system will automatically assign
    self.parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # Whether to use gradient accumulation
    self.parser.add_argument('--gn', action='store_true', help="Whether to use gradient accumulation")
    # Whether to debug
    self.parser.add_argument('--debug', action='store_true', help="Whether to use debug")

  def parse(self):
    self.opt = self.parser.parse_args()


    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt
