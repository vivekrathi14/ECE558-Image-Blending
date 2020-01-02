# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:07:51 2019

@author: Vivek Rathi
"""

import numpy as np
#import cv2
#import time

#padding function
def padr(img,t,b,l,r):
    ir = img.shape[0]
    ic = img.shape[1]
    if len(img.shape) == 3:
        imgpad = np.zeros((ir+t+b,ic+l+r,3), dtype = 'uint8')
    elif len(img.shape) == 2:
        imgpad = np.zeros((ir+t+b,ic+l+r), dtype = 'uint8')
    imgpad[t:ir+b,l:ic+r] = img
    rows = imgpad.shape[0]
    cols = imgpad.shape[1]
    for i,j in zip(range(t-1,-1,-1),range(t,t+t,1)):
        imgpad[i,:] = imgpad[j,:]
    
    for i,j in zip(range(l-1,-1,-1),range(l,l+l,1)):
        imgpad[:,i] = imgpad[:,j]
    
    for i,j in zip(range(rows-b,rows,1),range(rows-b-1,rows-b-b-1,-1)):
        imgpad[i,:] = imgpad[j,:]

    for i,j in zip(range(cols-r,cols,1),range(cols-r-1,cols-l-l-1,-1)):
        imgpad[:,i] = imgpad[:,j]
        
    return imgpad

def DFT2(img,size=0):
    if (img.dtype != complex):
        fimg = img.astype(np.float)
    else:
        fimg = np.copy(img)
    if size == 0:
        fftx = np.fft.fft(fimg,axis=0)
        fftxy = np.fft.fft(fftx,axis=1)
        return fftxy
    elif size != 0:
        fftx = np.fft.fft(fimg,size,axis=0)
        fftxy = np.fft.fft(fftx,size,axis=1)
        return fftxy
    
def IDFT2(fft2d):
    f_conj = np.conj(fft2d)
    ffxy = DFT2(f_conj,0)
    ffxy = ffxy / (fft2d.shape[0]*fft2d.shape[1])
    im = np.conj(ffxy)
    return im


def convfft(img,k):
    csize = img.shape[0] + k.shape[0] - 1
    pad_size = (csize - img.shape[0])//2
    t=b=l=r = pad_size
    if len(img.shape) == 3:
        k3 = np.zeros((k.shape[0],k.shape[1],3))
        k3[:,:,0] = k
        k3[:,:,1] = k
        k3[:,:,2] = k
    else:
        k3 = k
        
    imgpad = padr(img,t,b,l,r)
    img_f = DFT2(imgpad)
    k_f = DFT2(k3,imgpad.shape[0])
    p = np.multiply(img_f,k_f)
    inv_f = IDFT2(p)
    mag_i = abs(inv_f)
    mag_i = np.round(mag_i[2*pad_size:,2*pad_size:],5)
#    mag_i = np.round(mag_i[2*pad_size:,2*pad_size:],0)
    return np.float32(mag_i)

#img = cv2.imread('lena.png',1)
## Gaussian Kernel
#x = cv2.getGaussianKernel(3,2)
#k = x*x.T
#
##Impulse Kernel
#n = np.zeros((5,5))
#n[2,2] = 1.0
#
#
#flt = cv2.filter2D(img,-1,k)
#start = time.time()
#c2 = convfft(img,k)
#stop = time.time()
#print("Time- ",stop-start)

