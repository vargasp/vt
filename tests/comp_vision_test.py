#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 07:39:05 2022

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float


import vt.comp_vision as cv


def checker(M, N, p=5):
    E = np.tile(np.repeat([1,0],p),N*p).reshape(p,N*p*2)
    O = np.tile(np.repeat([0,1],p),N*p).reshape(p,N*p*2)
    
    return np.row_stack(M*(E, O))


img = img_as_float(data.chelsea())
img = np.flipud(img)


I = checker(5,10, p=50)
img = I

nX, nY = img.shape[:2]
c = [nX/2.0, nY/2]
c[0] = 0
t = [0.0, 0.0, 0.0]
t[0] = -250
theta = 0
phi = 0.0
psi = 60
s = (1.6, 1.6)
f = (1, 1)
mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi/nX,c=c,s=s,f=f)
print(mat)

img_t = cv.warp(img, mat)
plt.imshow(img_t, origin='lower')








img = img_as_float(data.chelsea())
img = np.flipud(img)

c = np.array(img.shape)[:2]/2.0
c[0] =0
t = (-100, 0.0, 0.0)
theta = 0
phi = 0.0
psi = 60
s = (1.0, 1.21)

mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi/nX,c=c,s=s,f=f)
print(mat)

img_t = cv.warp(img, mat)

plt.imshow(img_t, origin='lower')





"""
tform = transform.ProjectiveTransform(matrix=mat)
tf_img = transform.warp(img, tform.inverse)
fig, ax = plt.subplots()
ax.imshow(tf_img)
ax.set_title('Projective transformation 3x3')
plt.show()


mat = cv.euler_to_rot3(t=t,theta=theta,phi=phi,psi=psi,c=c,f=f)
print(mat)

tform = transform.ProjectiveTransform(matrix=mat)
tf_img = transform.warp(img, tform.inverse)
fig, ax = plt.subplots()
ax.imshow(tf_img[:,100:,:])
ax.set_title('Projective transformation 4x4')
plt.show()

"""