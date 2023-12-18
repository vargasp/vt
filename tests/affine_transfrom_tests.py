# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:14:44 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af
from skimage.transform import radon


def wobble(coords, center, angs, theta, phi):
    """
    [nAngles,nRows,nCols]

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.
    center : TYPE
        DESCRIPTION.
    angs : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    phi : TYPE
        Angle between the principle axis of stage rotation and sample rotation

    Returns
    -------
    None.

    """
    
    phis = np.arccos(np.cos(angs+theta))/np.pi * phi
    thetas = np.arccos(np.sin(angs+theta))/np.pi * phi


    for i, ang in enumerate(angs):
        R = af.rotateMat((0,phis[i],thetas[i]), center=center)
        RC = R @ coords[i,...]
        coords[i,...] = RC
        
    return coords
    
    

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
coords[:,:,2,0] = 32

T = af.transMat((64,64,0))
TC = (T @ coords)
test = af.coords_transform(phantom, TC)
plt.imshow(test[:,:,0],origin='lower')


R = af.rotateMat((0,0,90), center=np.array(phantom.shape)/2.0)
RC = (R @ coords)
test = af.coords_transform(phantom, RC)
plt.imshow(test[:,:,0],origin='lower')



coords = af.coords_array((nX,nY,1), ones=True)

T = af.transMat((0,0,32),rank=None)
R = af.rotateMat((0,90,0), center=np.array(phantom.shape)/2.0)
RTC = (R @ T @ coords)
test = af.coords_transform(phantom, RTC)
plt.imshow(test[:,:,0],origin='lower')






