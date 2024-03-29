#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:25:36 2022

@author: vargasp
"""

import numpy as np
import vt



i = np.meshgrid(range(21),range(11))[0]

vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title")
vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title",ticks=True)

vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title",ctitle="C Title")
vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title",ctitle="C Title",\
          ticks=True)

vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title",ctitle="C Title",\
          xlim=(0,210))
vt.imshow(i, title='Title', xtitle="X Title", ytitle="Y Title",ctitle="C Title",\
          xlim=(0,210),ylim=(-5,5))


    
i = np.meshgrid(range(21),range(11))[0]
vt.imshow2(i, title='Title')
    
    
    
    

    
vt.CreateImage(i, title='Title', xtitle="X Title", ytitle="Y Title")


vt.CreateImage(i, title='Title', xtitle="X Title", ytitle="Y Title",ctitle="ctitle")



vt.CreateImage(i, title='Title', xtitle="X Title", ytitle="Y Title", coords=[0,1,0,1],ctitle="ctitle")








x = np.linspace(100,200)
y = x**2
ys = np.array([y,1.1*y,1.2*y]).T
err_y = np.linspace(1000,2000)

"""

vt.CreatePlot(y, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')
vt.CreatePlot(ys, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')

vt.CreatePlot(y,err=err_y, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')
vt.CreatePlot(ys,err=err_y, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')

vt.CreatePlot(y,xs=x, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')
vt.CreatePlot(ys,xs=x, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')
vt.CreatePlot(y,xs=x,err=err_y, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')
vt.CreatePlot(ys,xs=x,err=err_y, title='Title Test',xtitle='X-Title',ytitle='Y-title', labels='Here')



vt.CreateBoxPlot(ys,title='Title Test',xtitle='X-Title',ytitle='Y-title', labels=['Here','t','y'])

"""

ys = np.zeros([2,50])
ys[0,:] = x
ys[1,:] = y


i = np.meshgrid(range(100),range(100))[0]
vt.CreateImage(i, title='Titled', xtitle="X Title", ytitle="Y Title", coords=[0,1,0,1],ctitle="ctitle")



i = np.meshgrid(range(20),range(20))[0]
vt.hist_show(i,bins=(5,5), window=(0,20), title='Title Main', xtitle="X Title", ytitle="Y Title", \
             ctitle="ctitle", xlabels=np.arange(5), ylabels=np.arange(5)+5)





#CreateImage(image,window=False,title ="",xtitle="",ytitle="",ctitle="",coords=None,outfile=None):
 