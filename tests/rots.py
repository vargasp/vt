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

def rankIdn(A, rank):
    n,m = A.shape
    
    if rank > n or rank >m:
        I = np.identity(rank)
        n,m  = min(n,rank), min(m,rank)
        I[:n,:m] = A[:n,:m]
        return I
    else:
        return A[:rank,:rank]





def transMat(coords,rank=None):
    coords = np.array(coords)
    n = coords.size
    
    if rank == None:
        rank = n+1
    
    T = np.identity(rank)
    T[:n,-1] = coords
    
    return T


def rotateMat(angs, center=None, seq='XYZ', extrinsic=True, rank=None):

    #Converts angs to an np.array and calcualtes the number of angles
    angs = np.array(angs)
    n = angs.size

    #
    if rank==None and n==1:
        rank = 2
    elif rank==None:
            

    #If the angle is intrinsic lower the sequence 
    if not extrinsic:
        seq = str.lower(seq)
    
    #Match the number of dimensions in the sequence to the number of angles
    seq = seq[(3-n):]
    
    #Calcuatd a 3x3 rotation matrix (centered at the 0,0)
    R =  transform.Rotation.from_euler(seq, angs, degrees=True).as_matrix()
    
    #Returns the R matrix or modifies it if rotation center is provided
    if center == None:
        return  R
    else:
        T = transMat(center, rank=4)
        R = rankIdn(R, 4)
        
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

