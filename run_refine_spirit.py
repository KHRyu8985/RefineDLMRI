import os, sys, h5py, torch, fastmri, bart, time
sys.path.append("train")
import numpy as np
import cupy as cp
import sigpy as sp

from train.modl import MoDLNet
from src.refine import RefineKspace

from src.evaluate import *
from fastmri.data import subsample, transforms
from src.evaluate import metrics, gmsd
import glob

def VCC(x):
    # x: kspace with (nC,nX,nY)
    vcc_ksp = x.copy()
    vcc_ksp = np.conj(vcc_ksp[:,::-1,::-1])
    if vcc_ksp.shape[1] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,1)
    if vcc_ksp.shape[2] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,2)
        
    return vcc_ksp

def espirit_calibrate(x, calsize=17):
    # x: kspace with (nC,nX,nY)
    # out: sens with (nC,nX,nY)
    tmp = np.moveaxis(x,0,-1)[:,:,None]
    senss = bart.bart(1,'ecalib -r {} -m 1 -P'.format(calsize),tmp)
    s = np.moveaxis(senss,-1,0).squeeze()
    return s

def DL_recon(ksp, mask, sens, net, option='ksp'):
    ksp = transforms.to_tensor(ksp)
    mask = transforms.to_tensor(mask[...,None]).byte()
    sens = transforms.to_tensor(sens)

    ksp = ksp.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    sens = sens.unsqueeze(0).cuda()
    
    with torch.no_grad():
        ksp_dl = net(ksp,mask,sens,option=option)
#        im_dl = net(ksp,mask, sens)
#    im_dl = transforms.tensor_to_complex_np(im_dl.cpu().detach()).squeeze().copy()
    
#    return im_dl
    ksp_dl = transforms.tensor_to_complex_np(ksp_dl.cpu().detach()).squeeze().copy()
    return ksp_dl

def runRefine(net, x_full, x_us, sens, calsize=21, lam=2, option='img'):
    # x_full: fully sampled kspace for ref

    
    # x_us: undersampled kspace with (nC,nX,nY)
    # First get sensitivity maps 
    if np.mean(np.abs(sens)) < 0.01:
        sens = espirit_calibrate(x_us,calsize)
    else:
        sens = sens
    im_full = sp.ifft(x_full, axes=(1,2))
    im_full = np.sum(im_full * np.conj(sens),0)
    
    mask = np.abs(x_us[0,100])>0
    mask = mask[None,None]
    if option == 'ksp':
        ksp_coil = DL_recon(x_us,mask,sens,net,option=option)
    else:
        im_dl = DL_recon(x_us,mask,sens,net,option=option)        
        im_coil = im_dl[None] * sens
        ksp_coil = sp.fft(im_coil, axes=(1,2))
    
    t0 = time.time()
    
    refine_op = RefineKspace(x_us.copy(),
                          ksp_coil.copy(),
                          method='spirit',
                          autoespirit=True,
                          nacs=calsize,
                          kSize=[5,5],
                          rank=200,
                          lam=lam,
                          vcc=False,
                          domain='conv',
                          verbose=True, device=sp.Device(0))

    ksp_refine = refine_op.run(niter=100, tol=1e-4)
    im_refine = np.sum(sp.ifft(ksp_refine, axes=(1,2)) * np.conj(sens), axis=0)

    print('Elapsed time:',time.time() - t0)
    if option == 'img':
        return im_full, im_dl, im_refine
    else:
        return x_full, ksp_coil, ksp_refine



def runspirit(subject_number, slice_number,lam=5, R=4, nacs=21, im_type='Brain', option='img'):

    # Loading model
    net = MoDLNet(num_cascades=10)
    if im_type == 'Brain':
        if R < 5:
            model_dir = 'train/models/Brain_T1Post_R4_MoDL/Brain_T1Post_R_4.pt'
        else:
            model_dir = 'train/models/Brain_T1Post_R6_MoDL/Brain_T1Post_R_6.pt'
        net.load_state_dict(torch.load(model_dir, map_location="cpu")['model_state_dict'])
    elif im_type == 'Knee':
        if R < 5:
            model_dir = 'train/models/fastMRI_mini_PD_R4_MoDL/fastMRI_mini_PD_R_4.pt'
        else:
            model_dir = 'train/models/fastMRI_mini_PD_R6_MoDL/fastMRI_mini_PD_R_6.pt'
        net.load_state_dict(torch.load(model_dir, map_location="cpu")['model_state_dict'])
    else:
        if R < 5:
            model_dir = 'train/models/fastMRI_mini_PD_FS_R4_MoDL/fastMRI_mini_PD_FS_R_4.pt'
        else:
            model_dir = 'train/models/fastMRI_mini_PD_FS_R6_MoDL/fastMRI_mini_PD_FS_R_6.pt'
        net.load_state_dict(torch.load(model_dir, map_location="cpu")['model_state_dict'])

    
    net = net.cuda()

    # loading subject
    
    if im_type == 'Brain':
        basedir = '/mnt/dense/kanghyun/Brain_T1Post/Test/'
    elif im_type == 'Knee':
        basedir = '/mnt/dense/kanghyun/fastMRI_mini_PD/Test/'
    else:
        basedir = '/mnt/dense/kanghyun/fastMRI_mini_PD_FS/Test'
    h5file = glob.glob(os.path.join(basedir, '*.h5'))[subject_number]
    
    with h5py.File(h5file,'r') as hr:
        X = hr['kspace'][slice_number] # saving the array to numpy (RAM)
        sens = hr['esp_maps'][slice_number]
        scale_factor = 10 / dict(hr.attrs)['max']

    X *= scale_factor
    
    # loading mask func
    cf = nacs / X.shape[-1]
    mask_func = subsample.MagicMaskFractionFunc(
                                center_fractions=[cf],
                                accelerations=[R]) 

    seednum = 400 # R=4
    mask, _ = mask_func(list(X.shape) + [1], seed=seednum) # trick because undersampling is in axis -1
    mask = mask[...,0]
    mask = mask.numpy()
        
    X_us = X * mask
    
    im_full, im_dl, im_refine = runRefine(net, X, X_us, sens, lam=lam, option=option)
    
    if option == 'img':
        xsize = 320

        im_d = sp.resize(im_dl[::-1], (xsize,xsize))
        im_r = sp.resize(im_refine[::-1], (xsize,xsize))
        im_f = sp.resize(im_full[::-1], (xsize,xsize))

        ssim_dl, psnr_dl, nrmse_dl = metrics(np.abs(im_d), np.abs(im_f))
        ssim_refine, psnr_refine, nrmse_refine = metrics(np.abs(im_r), np.abs(im_f))

        print('SSIM DL:{:.3f}, SSIM_Refine:{:.3f}, PSNR_DL:{:.2f}, PSNR_Refine:{:.2f}'.format(ssim_dl, ssim_refine, psnr_dl, psnr_refine))
        return im_d, im_r, im_f
    
    else:
        return im_dl, im_refine, im_full