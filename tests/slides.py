# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:14:30 2022

@author: vargasp
"""



import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import vt.comp_vision as cv
import helical_functions as hf
plt.rcParams['figure.dpi'] = 600
import vt




proj_dirs = ['Argonne','September_2021','AAA448 Daphnia PTA',\
             'proj_1632923835_9600projs_100000us_adult_daphnia_PTA_AAA448_async_tile_save_14kV']

nViews = 9600
views_per_rev = 3200
    
clipX=188
clipZ=208

gains = hf.reads_gains(proj_dirs)
view = 1600
img1 = hf.read_proj_slice(proj_dirs, view, gains, nViews)[clipZ:-clipZ,clipX:-clipX]    
vt.CreateImage(img1)

    
view = 3200
img2 = np.fliplr(hf.read_proj_slice(proj_dirs, view, gains, nViews))[clipZ:-clipZ,clipX:-clipX]   
vt.CreateImage(img2)


nX, nY = img2.shape[:2]
c = [nX/2.0, nY/2]
c[0] = 0
t = [0.0, 0.0, 0.0]
theta = 0
phi = 0.0
psi = 60
s = (1., 1.)
f = (1, 1)
mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi/nX,c=c,s=s,f=f)
print(mat)
img_t = cv.warp(img2, mat)
vt.CreateImage(img_t)

mat = cv.hmat(t=t,theta=theta,phi=phi,psi=-psi/nX,c=c,s=s,f=f)
print(mat)
img_tr = cv.warp(img_t, mat)
vt.CreateImage(img_tr)



nX, nY = img1.shape[:2]
c = [nX/2.0, nY/2]
c[0] = 0
t = [0.0, 0.0, 0.0]
theta = 0
phi = 0.0
psi = 60
s = (1., 1.)
f = (1, 1)
mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi/nX,c=c,s=s,f=f)
print(mat)
img_t = cv.warp(img1, mat)
vt.CreateImage(img_t)


nX, nY = img1.shape[:2]
c = [nX/2.0, nY/2]
c[0] = 1800
t = [0.0, 0.0, 0.0]
t[0] = -1800
theta = 0
phi = 0.0
psi = 60
s = (1., 1.)
f = (1, 1)
mat = cv.hmat(t=t,theta=theta,phi=phi,psi=psi/nX,c=c,s=s,f=f)
print(mat)
img_t = cv.warp(img1, mat)
vt.CreateImage(img_t)

nX, nY = img1.shape[:2]
c = [nX/2.0, nY/2]
c[0] = 0
t = [0.0, 0.0, 0.0]
mat = cv.hmat(t=t,theta=theta,phi=phi,psi=-psi/nX,c=c,s=s,f=f)
print(mat)
img_tr = cv.warp(img_t, mat)
vt.CreateImage(img_tr)

