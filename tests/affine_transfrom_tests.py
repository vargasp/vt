# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:14:44 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af
from skimage.transform import radon
from scipy.ndimage import affine_transform


def affine3d(arr, mat):
    return affine_transform(arr,mat,order=1,cval=0.0)

    
    
nX = 128
nY = 128
nZ = 64

phantom = np.zeros([nX, nY])
phantom[32:96,32:96] = 1
phantom = np.tile(phantom, (nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(nZ)


"""
2d
"""
coords = af.coords_array((nX,nY,1), ones=True)
coords[:,:,2,:] = 32
test = af.coords_transform(phantom, coords)
plt.imshow(test[:,:,0],origin='lower')


T = af.transMat((64,64,0))
TC = (T @ coords)
test = af.coords_transform(phantom, TC)
plt.imshow(test[:,:,0],origin='lower')


R = af.rotateMat((0,45,0), center=np.array(phantom.shape)/2.0)
RC = (R @ coords)
test = af.coords_transform(phantom, RC)
plt.imshow(test[:,:,0],origin='lower')


coords = af.coords_array((nX,nY,1), ones=True)
T = af.transMat((0,0,32),rank=None)
R = af.rotateMat((0,45,0), center=np.array(phantom.shape)/2.0)
RTC = (R @ T @ coords)
test = af.coords_transform(phantom, RTC)
plt.imshow(test[:,:,0],origin='lower')


"""
3d
"""

phantom = np.zeros([nX, nY])
phantom[56:72,56:72] = 1
phantom = np.tile(phantom, (nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(nZ)

coords = af.coords_array((nX,nY,nZ), ones=True)

R = af.rotateMat((0,0,90), center=np.array(phantom.shape)/2.0-.5)
RC = (R @ coords)
test = af.coords_transform(phantom, np.round(RC,6))
(phantom - test).max()

nAng = 361
angs = np.linspace(0,360,nAng,endpoint=True)
sino0 = np.zeros([nAng,nZ,nX])
sino10 = np.zeros([nAng,nZ,nX])
for i, ang in enumerate(angs):
    R = af.rotateMat((0,0,ang), center=np.array(phantom.shape)/2.0-.5)
    RC = (R @ coords)
    sino0[i,:,:] = af.coords_transform(phantom, np.round(RC,6)).sum(axis=1).T
    
    R = af.rotateMat((0,10,ang), center=np.array(phantom.shape)/2.0-.5)
    RC = (R @ coords)
    sino10[i,:,:] = af.coords_transform(phantom, np.round(RC,6)).sum(axis=1).T
    

plt.imshow(sino0[0,:,:], origin='lower')
plt.imshow(sino10[0,:,:], origin='lower')


"""
3d Coorection
"""
nAng = 361
coords = af.coords_array((nAng,nZ,nX), ones=True)
sino10f = np.zeros([nAng,nZ,nX])

R = af.rotateMat((0,0,5), center=np.array(phantom.shape)/2.0-.5)
RC = (R @ coords)


plt.imshow(sino10[45,:,:], origin='lower')

for i, ang in enumerate(angs):
    R = af.rotateMat((0,0,ang), center=np.array(phantom.shape)/2.0-.5)
    RC = (R @ coords)
    sino0[i,:,:] = af.coords_transform(phantom, np.round(RC,6)).sum(axis=1).T
    
    R = af.rotateMat((0,10,ang), center=np.array(phantom.shape)/2.0-.5)
    RC = (R @ coords)
    sino10[i,:,:] = af.coords_transform(phantom, np.round(RC,6)).sum(axis=1).T
    











"""
3D Warping
"""

R = af.rotateMat((0.,0.,90.), center=(nX/2, nY/2,0))
P = affine3d(phantom, R)
plt.imshow(P[54:74,54:74,32].T - phantom[54:74,54:74,32].T, origin='lower')




plt.imshow(P[:,64,:].T - phantom[:,64,:].astype(np.float32).T, origin='lower')



nAng = 361
angs = np.linspace(0,360,nAng,endpoint=True)
sino0 = np.zeros([nAng,nZ,nX])
sino10 = np.zeros([nAng,nZ,nX])
for i, ang in enumerate(angs):
    R = af.rotateMat((0,0,ang), center=(nX/2, nY/2,0))
    sino0[i,:,:] = affine3d(phantom, R).sum(axis=1).T
    R = af.rotateMat((0,10,ang), center=(nX/2, nY/2,0))
    sino10[i,:,:] = affine3d(phantom, R).sum(axis=1).T
    
    
    
plt.imshow(P, origin='lower')


plt.imshow(P[:,int(nY/2),:].T, origin='lower')









#Wooble
coords = af.coords_array((nAng,1,nX), ones=True)
coords[:,:,1,:] = 32
w_cords = wobble(coords, (nAng/2,nZ/2,nX/2), angs/180*np.pi, 35, 15)
test = af.coords_transform(sino, coords)
plt.imshow(test[:,0,:],origin='lower')



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
        R = af.rotateMat((phis[i],0, thetas[i]), center=center)
        RC = R @ coords[i,...]
        coords[i,...] = RC
        
    return coords
