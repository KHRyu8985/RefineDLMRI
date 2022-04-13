# RefineDLMRI

## Improving high frequency image features of Deep Learning reconstructions via k-space refinement with null-space kernel

Accepted in Magnetic Resonance in Medicine (2022)

March 19. 2022 Kanghyun Ryu (kanghyun@stanford.edu)

This repo contains code for Refining DL reconstruction via null-space kernel. 

![Refinement](https://github.com/KHRyu8985/RefineDLMRI/blob/main/Fig1_revised.png)

As can be seen in the Figure, (a) is undersampled MRI reconstruction from UNN (Unrolled Neural Network) and (b) is the proposed refinement process.

By utilizing a null-space kernel, we can correct for errors in Deep Learning's kspace estimates and refine reconstruction. 

Please refer to `demo.ipynb` for demonstration of running the code and examples.
