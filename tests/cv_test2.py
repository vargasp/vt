#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:04:35 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt

import vt.comp_vision as cv
plt.rcParams['figure.dpi'] = 600


def img_phantoms():
    img1 = np.zeros((200,100))
    img1[150,45:55] = 1.0
    img1[145:155,75] = 1.0
    img1[150,70:80] = 1.0
    img1[145:155,50] = 1.0
    img1[125,45:55] = 1.0
    img1[120:130,50] = 1.0

    img2 = np.zeros((200,100))
    img2[5::10,:] = 1.0
    img2[:,5::10] = 1.0

    return img1, img2


def warp_image(img,dX=0,dZ=0,theta=0.0,cZ=100,row=None):
    nZ, nX = img.shape

    cX = nX/2.0 + dX/2.0

    if row:
        dX = dX - np.sin(theta/180*np.pi)*(row - cZ)*2
        print(dX)
    
    mat = cv.hmat(t=(dZ, dX/-2.0),theta=theta,c=(cZ,cX))
    return cv.warp(img, mat)


img1, img2 = img_phantoms()
img3 = np.fliplr(img1)

img1t = warp_image(img1,dX=0,dZ=0,theta=5,cZ=100, row = 150)
img3t = warp_image(img1,dX=0,dZ=0,theta=-5,cZ=100, row = 150)


plt.imshow(img1 + img3, origin='lower')
plt.imshow(img1t + img3t, origin='lower')


plt.imshow(img2, origin='lower')





img2t = warp_image(img2,dX=50,dZ=0,theta=0,cZ=100)
plt.imshow(img2t, origin='lower')


img2 = warp_image(img1,dX=0,dZ=0,theta=.1,cZ=1000000., row=150)
plt.imshow(img2, origin='lower')


