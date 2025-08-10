import torch
import argparse
import os
from run.model_AU_Net import AU_Net_Cls
from run.dataset import TestData, imsave
from time import time
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES']='5'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='Reg&Fusion', help='Reg for only image registration, Fusion for only image fusion, Reg&Fusion for image registration and fusion')
parser.add_argument('--dataset_path', type=str, default='./datasets/test', help='The root directory for the dataset')
parser.add_argument('--dataset_name', type=str, default='Road', help='The name of the dataset')
parser.add_argument('--model_path', type=str, default='../checkpoint/ours/AU-Net.pth', help='Model path')
parser.add_argument('--gn', type=str, default='not_gn', help='Gradient Accumulation')
parser.add_argument('--first_conv_class', type=str, default='ODConv',
                         help="please choice the first conv class from ['static', 'deformable', 'dynamic', 'd2Conv', 'ODConv']")
parser.add_argument('--reg_conv_class', type=str, default='static',
                         help="please choice the reg_conv_class from ['static', 'deformable']")
parser.add_argument('--use_ir', action='store_true', help="Set whether to use the original image")
parser.add_argument('--save_root', type=str, default='../results', help='Root path to save results')
parser.add_argument('--save_path', type=str, default='Reg_Fusion_warp', help='Subdirectory to save the results')
parser.add_argument('--method_name', type=str, default='AU-Net', help='The name of the method')

if __name__ == '__main__':
    opts = parser.parse_args()
    img_path = os.path.join(opts.dataset_path, opts.dataset_name)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if opts.mode == 'Fusion':
        ir_path = os.path.join(img_path, 'ir')
    else:
        if opts.use_ir:
            ir_path = os.path.join(img_path, 'ir')
        else:
            ir_path = os.path.join(img_path, 'ir_warp')

    vi_path = os.path.join(img_path, 'vi')
    model_path = opts.model_path

    save_dir = os.path.join(opts.save_root, opts.save_path, opts.dataset_name, opts.method_name + '_' + opts.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    model = AU_Net_Cls(opts=opts)
    # # Whether to use Gradient Accumulation
    if opts.gn == 'gn':
        model.convert_bn_to_gn()
    model.resume(model_path, device)
    # model = model.cuda()
    model = model.to(device)
    model.eval()
    test_dataloader = TestData(ir_path, vi_path)
    test_dataloader_number = len(test_dataloader)
    p_bar = tqdm(enumerate(test_dataloader), total=test_dataloader_number)
    time_avg = 0.
    for idx, [ir, vi, name] in p_bar:
        # vi_tensor = vi.cuda()
        # ir_tenor = ir.cuda()

        vi_tensor = vi.to(device)
        ir_tenor = ir.to(device)

        start = time()
        with torch.no_grad():
            if opts.mode == 'Reg':
                results = model.registration_forward(ir_tenor, vi_tensor)
            elif opts.mode == 'Fusion':
                results = model.fusion_forward(ir_tenor, vi_tensor)
            else:
                results = model.forward(ir_tenor, vi_tensor)
        end = time()
        time_avg += (end - start) / test_dataloader_number
        imsave(results, os.path.join(save_dir, name))

        if opts.mode == 'Reg':
            p_bar.set_description(f'registering {name} | time : {str(round(end - start, 4))}')
        elif opts.mode == 'Fusion':
            p_bar.set_description(f'fusing {name} | time : {str(round(end - start, 4))}')
        else:
            p_bar.set_description(f'registering and fusing {name} | time : {str(round(end - start, 4))}')


    if opts.mode == 'Reg':
        print(f'registering  | time : {str(round(time_avg, 4))} seconds')
    elif opts.mode == 'Fusion':
        print(f'fusing  | time : {str(round(time_avg, 4))} seconds')
    else:
        print(f'registering and fusing  | time : {str(round(time_avg, 4))} seconds')
