""" Training E2E VarNet (Initial Simple Version) """
from dataset import MRIData
import matplotlib.pyplot as plt
from varnet import VarNet
from torch.utils.data import DataLoader
from sigpy import from_pytorch
import sigpy as sp
from torch.autograd import Variable
import logging
import json

# from vlogging import VisualRecord
from logging import FileHandler
import numpy as np
import torch
import argparse
import os
import sys
import logging

import skimage.metrics
from fastmri.data import transforms

from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.getcwd())

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def create_argparser():

    parser = argparse.ArgumentParser(
        description='Training  VarNet Reconstruction')

    parser.add_argument(
        '--datadir', default='/mnt/dense/kanghyun/Brain_T1Post',
        help='data root directory; where are datasets contained'
    )
    parser.add_argument('--num-epochs', type=int,
                        default=200, help='Number of Epochs')

    parser.add_argument('--R', type=int,
                        default=4, help='Acceleration (Defult=4')

    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')

    parser.add_argument(
        '--experimentname', default='unnamed_experiment',
        help='experiment name i.e. T1Post_R4?'
    )

    # save / display data
    parser.add_argument(
        '--verbose', default=1, type=int,
        help='''if true, prints to console average costs / metrics'''
    )

    parser.add_argument(
        '--tensorboard', default=1, type=int,
        help='if true, creates TensorBoard'
    )

    parser.add_argument(
        '--savefreq', default=10, type=int,
        help='how many epochs per saved recon image'
    )

    return parser


def train_epoch(epoch, data_loader, net, optimizer, criterion):
    cost = np.zeros(4)
    avg_loss = 0.
    for iter, (ksp, mask, sens, im_zf, im_fs) in enumerate(data_loader):

        ksp, mask, sens, im_fs = Variable(ksp, requires_grad=True).cuda(), Variable(mask).cuda(), Variable(
            sens, requires_grad=True).cuda(), Variable(im_fs, requires_grad=True).cuda()

        optimizer.zero_grad()
        im_dl = net(ksp, mask, sens)
#        im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
        loss = criterion(im_dl, im_fs) 
        loss.backward() 
        optimizer.step() 
        
        cost[0] += loss.item() 
        
        metric_val = metrics(im_fs, im_dl) # l1, ssim, psnr, nrmse
        for j in range(3):
            cost[j+1] += metric_val[j]
        
        avg_loss = 0.95 * avg_loss + 0.05 * loss.item() if iter > 0 else loss.item()

    cost = cost / len(data_loader)
    
    return cost


def validate_epoch(data_loader, net, criterion):

    cost = np.zeros(4)

    with torch.no_grad():
        for iter, (ksp, mask, sens, im_zf, im_fs) in enumerate(data_loader):
            ksp, mask, sens, im_fs = Variable(ksp, requires_grad=True).cuda(), Variable(mask).cuda(
            ), Variable(sens, requires_grad=True).cuda(), Variable(im_fs, requires_grad=True).cuda()
            im_dl = net(ksp, mask, sens)
#            im_us = transforms.complex_center_crop(
#                im_us, tuple(im_fs.shape[2:4]))

            loss = criterion(im_dl, im_fs)
            cost[0] += loss.item()
            metric_val = metrics(im_fs, im_dl) # l1, ssim, psnr, nrmse
            for j in range(3):
                cost[j+1] += metric_val[j]
        
        cost = cost / len(data_loader)

    return cost


def test_result(idx, testset, net):
    ksp, mask, sens, im_zf, im_fs = testset[idx]

    ksp = ksp.unsqueeze(0).cuda()
    im_fs = im_fs.unsqueeze(0)
    mask = mask.unsqueeze(0).cuda()
    sens = sens.unsqueeze(0).cuda()

    with torch.no_grad():
        im_dl = net(ksp, mask, sens)
#    im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))

    im_zf_np = from_pytorch(im_zf, iscomplex=True)
    im_dl_np = from_pytorch(im_dl[0].cpu().detach(), iscomplex=True)
    im_fs_np = from_pytorch(im_fs[0], iscomplex=True)

    out_cat = np.concatenate(
        (sp.resize(np.abs(im_zf_np).squeeze()[::-1], (320, 320)), sp.resize(np.abs(im_dl_np).squeeze()[::-1], (320, 320)), sp.resize(np.abs(im_fs_np).squeeze()[::-1], (320, 320))), 1)

    error_cat = np.concatenate(
        (sp.resize(np.abs(im_fs_np).squeeze()[::-1], (320, 320)), sp.resize(np.abs(im_fs_np).squeeze()[::-1], (320, 320)), sp.resize(np.abs(im_fs_np).squeeze()[::-1], (320, 320))), 1)

    error_cat = np.abs(error_cat - out_cat) * 5

    out_cat = np.concatenate((out_cat, error_cat), axis=0)
    out_cat = out_cat * 10
    return out_cat


def main(args, writer, model_name):
    """ Creating a masking function """
    dset_name = args.datadir
    
    save_name = '/home/kanghyun/projRSC/ESPIRIT_RSC/train/logs/train_' + \
        dset_name.split('/')[-1] + '_R_' + str(args.R) + '.txt'

    fh = FileHandler(save_name, mode="w")
    logger.addHandler(fh)
    if args.verbose:
        logger.setLevel(logging.INFO) 

    net = VarNet(num_cascades=12)
    net = net.cuda()

    criterion = torch.nn.L1Loss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    trainset = MRIData([dset_name + '/Train'], R=args.R)
    validset = MRIData([dset_name + '/Val'], R=args.R, test=True)
    testset = MRIData([dset_name + '/Test'], R=args.R, test=True)

    trainloader = DataLoader(trainset, batch_size=1,
                             shuffle=True, num_workers=8)

    validloader = DataLoader(validset, batch_size=1,
                             shuffle=True, num_workers=8)

    logger.info('Training .... ')

    best_val_ssim = 0
    for epoch in range(args.num_epochs):
        train_cost = train_epoch(
            epoch, trainloader, net, optimizer, criterion)
        logger.info('Epoch: {}, Train Loss: {}, Train SSIM: {}'.format(epoch, train_cost[0], train_cost[1]))

        val_cost = validate_epoch(
            validloader, net, criterion)
        logger.info('Epoch: {}, Val Loss: {}, Val SSIM: {}'.format(epoch, val_cost[0], val_cost[1]))

        if args.tensorboard and epoch > 0:
            write_tensorboard(writer, train_cost, val_cost, epoch)
        
        if epoch % args.savefreq == 0:
            for idx in [2, -70]:
                out_cat = test_result(idx, testset, net)

                fig = plt.figure(dpi=300)
                plt.imshow(np.abs(out_cat), cmap='gray', vmax=50)
                plt.axis('off')
                # logger.debug(VisualRecord(
                #     "epoch: {}, slice:{}".format(epoch, idx), fig, fmt="png"))                
                if args.tensorboard:
                    writer.add_figure(f'Results slice: {idx}', fig, epoch, close=True)
                plt.close(fig)

            if best_val_ssim < val_cost[1]:            
                # Save network to weight
                weight_name = model_name + \
                    opt.datadir.split('/')[-1] + '_R_' + str(args.R) + '.pt'
                torch.save({'epoch':epoch, 'model_state_dict': net.state_dict(),'val_ssim':val_cost[1]}, weight_name)


def metrics(im_fs: torch.Tensor, im_us: torch.Tensor):
    '''
    @parameter im_us: undersampled image (2D)
    @parameter im_fs: fully sampled image (2D)
    should be on GPU device for fast computation
    '''

    # change to ndarray
    im_us = transforms.tensor_to_complex_np(im_us.cpu().detach())
    im_fs = transforms.tensor_to_complex_np(im_fs.cpu().detach())
    
    # convert complex nums to magnitude
    im_us = np.absolute(im_us)
    im_fs = np.absolute(im_fs)
    
    im_us = im_us.reshape(
        (im_us.shape[2], im_us.shape[3])
    )
    
    im_fs = im_fs.reshape(
        (im_fs.shape[2], im_fs.shape[3])
    )
    
    # psnr
    psnr = skimage.metrics.peak_signal_noise_ratio(
        im_fs, 
        im_us, 
        data_range = np.max(im_fs) - np.min(im_fs)
    )
    
    #nrmse
    nrmse = skimage.metrics.normalized_root_mse(im_fs, im_us)
    
    # ssim
    # normalize 0 to 1
    im_fs -= np.min(im_fs)
    im_fs /= np.max(im_fs)
    im_us -= np.min(im_us)
    im_us /= np.max(im_us)
    
    ssim = skimage.metrics.structural_similarity(im_fs, im_us, data_range = 1)
    
    return ssim, psnr, nrmse

def write_tensorboard(writer, train_cost, val_cost, iteration):
        
    writer.add_scalars(
        f'l1', {
            f'train' : train_cost[0],
            f'val' : val_cost[0],
        }, 
        iteration
    )

    writer.add_scalars(
        f'ssim', {
            f'train' : train_cost[1],
            f'val' : val_cost[1],
        }, 
        iteration
    )

    writer.add_scalars(
        f'psnr', {
            f'train' : train_cost[2],
            f'val' : val_cost[2],
        }, 
        iteration
    )

    writer.add_scalars(
        f'nrmse', {
            f'train' : train_cost[3],
            f'val' : val_cost[3],
        }, 
        iteration
    )

if __name__ == "__main__":
    opt = create_argparser().parse_args()
    run_name = f"/home/kanghyun/projRSC/ESPIRIT_RSC/train/runs/{opt.experimentname}/"
    model_name = f"/home/kanghyun/projRSC/ESPIRIT_RSC/train/models/{opt.experimentname}/"
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
    writer_tensorboard = SummaryWriter(log_dir=run_name)

    logger = logging.getLogger('Training-VarNet')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # write json files to runs directory
    with open(os.path.join(run_name, 'parameters.json'), 'w') as parameter_file:
        json.dump(vars(opt), parameter_file)

    with open(os.path.join(model_name, 'parameters.json'), 'w') as parameter_file:
        json.dump(vars(opt), parameter_file)
    
    main(opt, writer_tensorboard, model_name)
    writer_tensorboard.flush()
    writer_tensorboard.close()