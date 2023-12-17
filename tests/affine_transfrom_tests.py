# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:14:44 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af
from skimage.transform import radon

nX = 128
nY = 128
nZ = 64
nAng = 180

phantom = np.zeros([nX, nY])
phantom[32:96,32:96] = 1
phantom = np.tile(phantom, (nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(nZ)

plt.imshow(phantom[:,int(nY/2),:].T, origin='lower')

angs= np.arange(nAng)

sino = np.zeros([nAng,nZ,nX])

for z in range(nZ):
    sino[:,z,:] = radon(phantom[:,:,z], theta=angs).T


plt.imshow(sino[:,int(nZ/2),:].T, origin='lower')


coords = af.coords_array((nX,nY,1), ones=True)
coords = np.transpose(coords,[1,2,0,3])
coords = np.ascontiguousarray(coords)
coords[:,:,2,0] = 32

T = af.transMat((64,64,0),rank=None)
TC = (T @ coords)
TC = np.transpose(TC,[2,0,1,3])[:3,...]
test = af.coords_transform(phantom, TC)
plt.imshow(test[:,:,0],origin='lower')


R = rotateMat((0,0,90), center=np.array(phantom.shape)/2.0)
RC = (R @ coords)
RC = np.transpose(RC,[2,0,1,3])[:3,...]
test = coords_transform(phantom, RC)
plt.imshow(test[:,:,0],origin='lower')



coords = coords_array((nX,nY,1), ones=True)
coords = np.transpose(coords,[1,2,0,3])
coords = np.ascontiguousarray(coords)

T = transMat((0,0,32),rank=None)
R = rotateMat((0,90,0), center=np.array(phantom.shape)/2.0)
RTC = (R @ T @ coords)
RTC = np.transpose(RTC,[2,0,1,3])[:3,...]
test = coords_transform(phantom, RTC)
plt.imshow(test[:,:,0],origin='lower')






