# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:30:07 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vt.comp_vision as cv
from scipy.spatial import transform
from scipy.ndimage import affine_transform


def transGenD(coords):
    coords = np.array(coords)
    n = coords.size
    T = np.identity(n+1)
    T[:n,n] = coords
    
    return T
    


def rotate2D(theta):
    return transform.Rotation.from_euler('z', [theta], degrees=True).as_matrix()


def rotate3D(theta, phi, psi, extrinsic=True):
    if extrinsic:
        return transform.Rotation.from_euler('XYZ', [psi,phi,theta], degrees=True).as_matrix()
    else:
        return transform.Rotation.from_euler('xyz', [psi,phi,theta], degrees=True).as_matrix()


def rotateGenD(angs, extrinsic=True):
    angs = np.array(angs)
    if angs.size == 1:
        return rotate2D(angs)
    else:
        return rotate3D(*angs, extrinsic=extrinsic)


def rotate(angs, center=None, extrinsic=True):
    
    if center == None:
        return rotateGenD(angs, extrinsic=extrinsic)
    else:
        T = transGenD(center)
        R  = rotateGenD(angs, extrinsic=extrinsic)
        return np.linalg.inv(T) @ R @ T


nX = 5
nY = 10
nZ = 15

X, Y = np.mgrid[:nX, :nY]
coords2 = np.stack([X,Y,np.ones(X.shape)], axis = -2, dtype=float)

X, Y, Z = np.mgrid[:nX, :nY, :nZ]
coords3 = np.stack([X,Y,Z,np.ones(X.shape)], axis = -2, dtype=float)


T = trans2D(1,-.1)
TC = (T @ coords)[:,:2,:]

