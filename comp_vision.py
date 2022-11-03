#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 07:36:40 2022

@author: vargasp
"""

import numpy as np

import cv2
import scipy.spatial.transform as transform
from scipy import ndimage



def sobel2d(img):
    """
    Creates a 2d Sobel filtered image
    
    Parameters
    ----------
    img : (N,M) array
        image to filter

    Returns
    -------
    (N,M) array
        The filtered image
    """
    
    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    return np.hypot(dx, dy)


def hmat(t=(0.0,0.0,0.0),theta=0.0,phi=0.0,psi=0.0,c=0.0,s=1.0):
    """
    Creates a homography matrix from angle rotations, translations, and scaling
    
    Parameters
    ----------
    t : scalar or array_like
        The translation vector. Default 0.0
    s : scalar or array_like
            The scaling vector (sx, sy). Default 1.0
    c : scalar or array_like
        The center of rotation vector (cx, cy). Default 0.0
    phi : float
        The pitch rotation angle in radians

    Returns
    -------
    R  (3,3) numpy array 
        The homogenous transforation array
    """
    
    c = np.array(c,dtype=float)
    if c.size == 1:
        c = np.repeat(c,3)
    elif c.size == 2:
        c = np.append(c,0.0)

    s = np.array(s,dtype=float)
    if s.size == 1:
        s = np.repeat(s,2)


    t = np.array(t,dtype=float)
    if t.size == 1:
        raise Exception("T: Translation vector must be in x,y or x,y,z coords")
    elif t.size == 2:
        t = np.append(t,0.0)
    
    #Intrinsic imaging matrix
    f=1
    I = np.identity(3)
    I[0,0] = f
    I[1,1] = f
    I[0,2] = -c[0]
    I[1,2] = -c[1]
        
    #Translation/Scaling matrix
    ST = np.identity(3)
    ST[0,0] = s[0]
    ST[1,1] = s[1]
    ST[0,2] = t[0]
    ST[1,2] = t[1]
    
    #Rotation Matrix
    R = transform.Rotation.from_euler('xyz', [psi,phi,theta], degrees=True).as_matrix()
    
    return np.linalg.inv(I) @ ST @ R @ I



def homgraphy_mat(img1, img2):

    # covert images to 8 bit for cv2 functions
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # define constants
    MIN_MATCH_COUNT = 10
    MIN_DIST_THRESHOLD = 0.7
    RANSAC_REPROJ_THRESHOLD = 5.0

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # find matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < MIN_DIST_THRESHOLD * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        return H

    else:
        raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    



def homgraphy_ma2t(img1, img2):

    # covert images to 8 bit for cv2 functions
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # define constants
    MIN_MATCH_COUNT = 10
    MIN_DIST_THRESHOLD = 0.7
    RANSAC_REPROJ_THRESHOLD = 5.0

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # find matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < MIN_DIST_THRESHOLD * n.distance:
            good.append(m)


    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        matchesMask = mask.ravel().tolist()
        
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        return cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    else:
        raise Exception("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    









def hmat4(t=(0.0,0.0,0.0),theta=0.0,phi=0.0,psi=0.0,c=0.0,f=1.0):
    """
    !!!!!!NOT IMPLEMNTED!!!!!
    
    
    Parameters
    ----------
    t : TYPE, optional
        DESCRIPTION. The default is (0.0,0.0,0.0).
    theta : TYPE, optional
        DESCRIPTION. The default is 0.0.
    phi : TYPE, optional
        DESCRIPTION. The default is 0.0.
    psi : TYPE, optional
        DESCRIPTION. The default is 0.0.
    c : TYPE, optional
        DESCRIPTION. The default is 0.0.
    f : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    

    c = np.array(c,dtype=float)
    if c.size == 1:
        c = np.repeat(c,3)
    elif c.size == 2:
        c = np.append(c,0.0)

    t = np.array(t,dtype=float)
    if t.size == 1:
        raise Exception("T: Translation vector must be in x,y or x,y,z coords")
    elif t.size == 2:
        t = np.append(t,0.0)
    
    """
    #Intrinsic imaging matrix
    I = np.eye(4,3)
    I[0,0] = f
    I[1,1] = f
    I[0,2] = -c[0]
    I[1,2] = -c[1]
    """

    #Intrinsic imaging matrix
    I = np.identity(3)
    I[0,0] = f
    I[1,1] = f
    I[0,2] = -c[0]
    I[1,2] = -c[1]



    #Translation matrix
    T = np.identity(4)
    T[0,3] = t[0]
    T[1,3] = t[1]
    T[2,3] = t[2]
    
    """
    RT = np.identity(4)
    RT[:3,:3] = transform.Rotation.from_euler('xyz', [psi,phi,theta], degrees=True).as_matrix()
    RT[:3,3] = t
    print(RT)
    """
    
    RT = np.identity(4)
    RT[[0,1,3],:3] = transform.Rotation.from_euler('xyz', [psi,phi,theta], degrees=True).as_matrix()
    
    print(RT)
    print(T)
    
    print(T @ RT)
    
    print(RT)
    
    e1 = np.eye(3,4)
    e2 = np.eye(4,3)[[0,1,3,2]]

    #print(np.linalg.inv(I) @ e1 @ T @ R @ e2 @ I)
    return np.linalg.inv(I) @ e1 @ RT  @ e2 @ I
    #print(R)
    #print(T @ R)
    """
    print(T)
    print(T @ R)
    print(np.linalg.pinv(I).shape)
    print(T.shape)
    print(R.shape)
    print(I.shape)
    """
   # return np.linalg.pinv(I) @ T @ R @ I

    