import pylab as py
import importlib

import numpy as np
import vt

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
 