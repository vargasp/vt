#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 07:39:05 2022

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform
from skimage import img_as_float

import comp_vision as cv

img = img_as_float(data.chelsea())



cx = 451./2
cy = 150
cz = 0
tx = 0
ty = 0
tz = 0
theta = 0
phi = 0.1
psi = 0.0
f = .10

t = (tx,ty,tz)
c = (cx,cy,cz)

img = img_as_float(data.chelsea())
mat = cv.euler_to_rot2(t=t,theta=theta,phi=phi,psi=psi,c=c,s=f)
print(mat)
print()

tform = transform.ProjectiveTransform(matrix=mat)
tf_img = transform.warp(img, tform.inverse)
fig, ax = plt.subplots()
ax.imshow(tf_img)
ax.set_title('Projective transformation 3x3')
plt.show()



"""
mat = cv.euler_to_rot3(t=t,theta=theta,phi=phi,psi=psi,c=c,f=f)
print(mat)

tform = transform.ProjectiveTransform(matrix=mat)
tf_img = transform.warp(img, tform.inverse)
fig, ax = plt.subplots()
ax.imshow(tf_img[:,100:,:])
ax.set_title('Projective transformation 4x4')
plt.show()

"""