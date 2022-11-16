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



img = img_as_float(data.chelsea())
img = np.flipud(img)

c = np.array(img.shape)[:2]/2.0
c[0] =0
t = (0.0, 0.0, 0.0)
theta = 0
phi = 0.0
psi = 0.1
s = (1.0, 1.0)

mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi,c=c,s=s)
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
psi = 0.1
s = (1.0, 1.21)

mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi,c=c,s=s)
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