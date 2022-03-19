import numpy as np
import h5py
import pathlib
from fastmri.data import subsample
from fastmri.data import transforms
from torch.utils.data import Dataset
import sigpy as sp
import fastmri
import torch
class MRIData(Dataset):
    """Generate masked kspace, mask, sens, zf_image, full_image = MRIData(root_dir, mask_func=None, subject_name=None, R=R, test=False) : Training
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, root, mask_func=None, subject_name=None, R=4, test=False):
        super().__init__()
        self.examples = []

        if subject_name == None and mask_func == None:
            # This is definitely training stage
            if isinstance(root, list):  # if multiple roots
                Files = []
                for tmp in root:
                    Files += list(pathlib.Path(tmp).glob('*.h5'))

            self.mask_func = subsample.EquispacedMaskFunc(
                center_fractions=[0.08, 0.085, 0.09, 0.095, 0.1, 0.15],
                accelerations=[R, R, R, R, R, R])

            self.test = False

        else:
            if subject_name:
                Files = list(pathlib.Path(root).glob(subject_name))
                assert len(Files) == 1, 'something wrong'

            if mask_func == None:  # Let's make it manual
                self.mask_func = subsample.EquispacedMaskFunc(
                    center_fractions=[0.08], accelerations=[R])

            self.test = True

        if test == True: # overwrite this
            self.mask_func = subsample.EquispacedMaskFunc(
                    center_fractions=[0.08], accelerations=[R])
            self.test = True  
                      
        for fname in sorted(Files):
            data = h5py.File(fname, 'r')
            ksp = data['kspace']

            num_slices = ksp.shape[0]
            self.examples += [(fname, slice_num)
                              for slice_num in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=0, keepdim=True
        )
    
    def __getitem__(self, idx):
        fname, sl = self.examples[idx]
        with h5py.File(fname, 'r') as hr:
            kspace, sens = hr['kspace'][sl], hr['esp_maps'][sl]
        kspace = kspace * 10000

        im_coil = sp.ifft(kspace, axes=[1, 2])
        im_comb = np.sum(im_coil * np.conj(sens), axis=0)
        if self.test:
            mask = self.mask_func(list(im_comb.shape) + [1], seed=700)[..., 0]
        else:
            mask = self.mask_func(list(im_comb.shape) + [1])[..., 0]
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask 
        mask = np.expand_dims(mask, axis=-1)

        masked_kspace = transforms.to_tensor(masked_kspace)
        kspace = transforms.to_tensor(kspace)
        
        mask = transforms.to_tensor(mask)
        sens = transforms.to_tensor(sens)
        
        zf_image = self.sens_reduce(masked_kspace, sens)
        full_image = self.sens_reduce(kspace, sens)
        
        return masked_kspace, mask.byte(), sens, zf_image, full_image # changed return values (added zf_image)

if __name__ == "__main__":
    from logging import FileHandler
    from vlogging import VisualRecord
    import logging
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger('Unit-testing-dataset')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - $(message)s'))
    logger.addHandler(handler)

    fh = FileHandler('Unit_testing_dataset.html', mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    
    dset_name = '/mnt/dense/kanghyun/fastMRI_mini_PD_FS/Train'
    trainset = MRIData(root=[dset_name])
    testset = MRIData(root=dset_name, subject_name='file1000000.h5')
    
    logger.info('Trainset length: {}'.format(len(trainset)))
    logger.info('Testset length: {}'.format(len(testset)))
    
    logger.info('Now see if mask is random for Trainset')
    
    masked_kspace1, mask1, sens1, zf_image1, full_image1 = trainset[20]
    masked_kspace2, mask2, sens2, zf_image2, full_image2 = trainset[10]
    
    mask1 = mask1.numpy().squeeze()
    mask2 = mask2.numpy().squeeze()
    
    full_image1 = transforms.tensor_to_complex_np(full_image1)
    zf_image1 = transforms.tensor_to_complex_np(zf_image1)
  
    full_image2 = transforms.tensor_to_complex_np(full_image2)
    zf_image2 = transforms.tensor_to_complex_np(zf_image2)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=100)
    ax1.plot(mask1[:50])
    ax1.set_title('mask1')
    ax2.plot(mask2[:50])
    ax2.set_title('mask2')
    fig.tight_layout()
    
    logger.debug(VisualRecord('Mask random for training',fig,fmt="png"))
    plt.close(fig)
    
    logger.info('See if input and ground truth is correct')
    fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=100)
    ax1.imshow(np.abs(full_image1).squeeze(), cmap='gray')
    ax1.set_title('Fully-sampled')
    ax2.imshow(np.abs(zf_image1).squeeze(), cmap='gray')
    ax2.set_title('Under-sampled')
    fig.tight_layout()

    logger.debug(VisualRecord('Input and Label for Train-set',fig,fmt="png"))
    plt.close(fig)    
    
    logger.info('Now see if mask is equal for Testset')

    masked_kspace1, mask1, sens1, zf_image1, full_image1 = trainset[20]
    masked_kspace2, mask2, sens2, zf_image2, full_image2 = trainset[10]
    
    mask1 = mask1.numpy().squeeze()
    mask2 = mask2.numpy().squeeze()  
      
    masked_kspace1 = transforms.tensor_to_complex_np(masked_kspace1)
    sens1 = transforms.tensor_to_complex_np(sens1)
    
    full_image1 = transforms.tensor_to_complex_np(full_image1)
    zf_image1 = transforms.tensor_to_complex_np(zf_image1)
  
    full_image2 = transforms.tensor_to_complex_np(full_image2)
    zf_image2 = transforms.tensor_to_complex_np(zf_image2)
                
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=100)
    ax1.plot(mask1[:50])
    ax1.set_title('mask1')
    ax2.plot(mask2[:50])
    ax2.set_title('mask2')
    fig.tight_layout()
    
    logger.debug(VisualRecord('Mask equal for testing',fig,fmt="png"))
    plt.close(fig)
    
    logger.info('See if input and ground truth is correct')
    fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=100)
    ax1.imshow(np.abs(full_image1).squeeze(), cmap='gray')
    ax1.set_title('Fully-sampled')
    ax2.imshow(np.abs(zf_image1).squeeze(), cmap='gray')
    ax2.set_title('Under-sampled')
    fig.tight_layout()
    
    logger.debug(VisualRecord('Input and Label for Test-set',fig,fmt="png"))
    plt.close(fig)
        
    mask_conj = mask1[::-1]
    mask_conj = np.roll(mask_conj,1,0)
    
    logger.info("See if VCC works")
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=100)
    ax1.plot(mask1[:50])
    ax1.set_title('mask1')
    ax2.plot(mask_conj[:50])
    ax2.set_title('mask1 conj')
    fig.tight_layout()
    
    logger.debug(VisualRecord('Mask conj for testing',fig,fmt="png"))
    plt.close(fig)
    

    
    
     

    
    
    
    
    
    
    
    
    
    
    