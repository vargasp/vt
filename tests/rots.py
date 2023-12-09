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


def transMat(coords,rank=None):
    coords = np.array(coords)
    n = coords.size
    
    if rank == None:
        rank = n+1
    
    T = np.identity(rank)
    T[:n,-1] = coords
    
    return T


def rotateMat(angs, center=None, seq='XYZ', extrinsic=True):
    if not extrinsic:
        seq = str.lower(seq)
    
    angs = np.array(angs)
    n = angs.size
    seq = seq[(3-n):]
        
    R =  transform.Rotation.from_euler(seq, angs, degrees=True).as_matrix()
    
    if center == None:
        return  R
    else:
        RN = np.identity(4)
        RN[:3,:3] = R
        R = RN
        
        T = transMat(center, rank=4)
        print(T.shape)
        print(R.shape)
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

