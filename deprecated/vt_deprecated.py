from os.path import expanduser
import pylab as py
import numpy as np
import matplotlib as mpl

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib.patches import Ellipse



def CreateImageCart(image,window=False,title ="",xtitle="",ytitle="",ctitle="",X=None,Y=None,coords=False,outfile=None):
    """
    Displays/Creates a BW intensity image

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    
    Xn, Yn = image.shape
    
    if window == False:
        window = [np.min(image),np.max(image)]
    
    extent = [0,0,0,0]
    if X is None:
        extent[1] = Xn-1
    else:
        extent[0] = X[0]
        extent[1] = X[-1]
        coords = True
        
    if Y is None:
        extent[3] = Yn-1
    else:
        extent[2] = Y[0]
        extent[3] = Y[-1]
        coords = True

    fig, ax = plt.subplots()
    #im = plt.imshow(image, vmin=window[0], vmax=window[1])
    im = plt.imshow(image.T, vmin=window[0], vmax=window[1], cmap=cm.Greys_r, \
            origin='lower', extent=extent)
    #ax.set_title(title, fontsize=14)

    

    if coords == False:
        ax.axis('off')

    if ctitle != "":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=window)
        cbar.ax.set_ylabel(ctitle)
        cbar.ax.tick_params(labelsize=10) 

    LabelPlot(ax,title,xtitle,ytitle)
    DisplayPlot(fig,outfile)


def CreateImages(Images,dims=None,window=False,title="",labels=None,ctitle="",outfile=None):
    
    Images = np.array(Images)
    nImages,xPixels,yPixels, = Images.shape
    
    if dims == None:
        nRows = np.ceil(np.sqrt(nImages)).astype(int)
        nCols = np.ceil(np.sqrt(nImages)).astype(int)
    else:
        nRows = dims[0].astype(int)
        nCols = dims[1].astype(int)
    
    
    indices = np.indices((nRows,nCols))
    i, j = np.meshgrid(range(nCols),range(nRows))
    i = i.flatten()
    j = j.flatten()

    labels = FormatLabels(labels, nImages)

    fig, ax = plt.subplots(nRows,nCols,figsize=(nCols*2,nRows*2))
    for n in range(nImages):
        if window == False:
            window = [np.min(Images[n,:,:]),np.max(Images[n,:,:])]
        
        ax[j[n], i[n]].imshow(Images[n,:,:], cmap=cm.Greys_r, vmin=window[0], vmax=window[1])
        ax[j[n], i[n]].set_title(labels[n])

    for n in range(nRows*nCols):
        ax[j[n], i[n]].axis('off')
    
    #plt.subplots_adjust(wspace=.1, hspace=.1)
    plt.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
    plt.subplots_adjust(left=0.0,right=1.0,top=1.0,bottom=0.0)
    DisplayPlot(fig,outfile)


def CreateBarChart(ys,x=None,title="",xtitle="",ytitle="",xticklabels="",labels=False,grid=False,outfile=False):    
    """
    Displays/Creates a plot with Mobius detector dimensions

    Parameters:        
        array:    1d or 2d array
        title:    title for the plot
        ylabel:   title for the y-axis
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    width = 0.35
        
    ys = FormatYs(ys)
    nPlots, nPnts = ys.shape
    xs = np.arange(nPnts)
    
    labels = FormatLabels(labels, nPlots)

                  
    fig, ax = plt.subplots()
    for i in range(nPlots):
        ax.bar(xs+width,ys[i,:],width)

    if labels[0] != '':
        plt.legend(loc=0, fontsize=12)
    
    ax.set_xticks(xs + 1.5*width)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(xticklabels, fontsize=12,fontname='Times New Roman')
    
    LabelPlot(ax,title,xtitle,ytitle)    
    DisplayPlot(fig,outfile)



def DrawEllipse(ep, ax, rgb_values,alpha_value):
    e = Ellipse((ep[0],ep[1]),width=ep[2],height=ep[3],angle=ep[4])
    ax.add_artist(e)
    e.set_alpha(alpha_value)
    e.set_facecolor(rgb_values)
    
def CreateEllipsesPlot(ell1,ell2,title='',xtitle='',ytitle='',labels=['',''],outfile=None):
    '''
    ell1: [5, nEllipses], [x,y,dx,dy,cor]
    ell2: [5, nEllipses], [x,y,dx,dy,cor]
    '''
   
    alpha_value = 0.4
    rgb_r = (.3,.1,.1)
    rgb_b = (.1,.1,.3)

    fig, ax = plt.subplots()
    plt.plot(ell1[0,:],ell1[1,:],'bo',label=labels[0])
    plt.plot(ell2[0,:],ell2[1,:],'ro',label=labels[1])
    aspect_ratio = (ax.axis()[1] - ax.axis()[0])/(ax.axis()[3] - ax.axis()[2])
    ell1[4,:] = 0.5*np.arctan(aspect_ratio*2.0*ell1[2,:]*ell1[3,:]*ell1[4,:]/(ell2[2,:]**2 - ell1[3,:]**2))/np.pi*180
    ell2[4,:] = 0.5*np.arctan(aspect_ratio*2.0*ell1[2,:]*ell1[3,:]*ell1[4,:]/(ell2[2,:]**2 - ell1[3,:]**2))/np.pi*180

    for i in range(len(ell1[0,:])):
        DrawEllipse(ell1[:,i], ax, rgb_b, alpha_value)

    for i in range(len(ell2[0,:])):
        DrawEllipse(ell2[:,i], ax, rgb_r, alpha_value)

    #ax.set_aspect('equal', 'datalim')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=2)

    DisplayPlot(fig,outfile)



def CreateEllipsesPlot3(ell1,ell2,ell3,title='',xtitle='',ytitle='',labels=['',''],outfile=None):
    '''
    ell1: [5, nEllipses], [x,y,dx,dy,cor]
    ell2: [5, nEllipses], [x,y,dx,dy,cor]
    '''
   
    alpha_value = 0.4
    rgb_r = (.3,.1,.1)
    rgb_b = (.1,.1,.3)
    rgb_g = (.1,.3,.1)


    fig, ax = plt.subplots()
    plt.plot(ell1[0,:],ell1[1,:],'bo',label=labels[0])
    plt.plot(ell2[0,:],ell2[1,:],'ro',label=labels[1])
    plt.plot(ell3[0,:],ell3[1,:],'go',label=labels[2])
    aspect_ratio = (ax.axis()[1] - ax.axis()[0])/(ax.axis()[3] - ax.axis()[2])
    ell1[4,:] = 0.5*np.arctan(aspect_ratio*2.0*ell1[2,:]*ell1[3,:]*ell1[4,:]/(ell2[2,:]**2 - ell1[3,:]**2))/np.pi*180
    ell2[4,:] = 0.5*np.arctan(aspect_ratio*2.0*ell1[2,:]*ell1[3,:]*ell1[4,:]/(ell2[2,:]**2 - ell1[3,:]**2))/np.pi*180
    ell3[4,:] = 0.5*np.arctan(aspect_ratio*2.0*ell1[2,:]*ell1[3,:]*ell1[4,:]/(ell2[2,:]**2 - ell1[3,:]**2))/np.pi*180


    for i in range(len(ell1[0,:])):
        DrawEllipse(ell1[:,i], ax, rgb_b, alpha_value)

    for i in range(len(ell2[0,:])):
        DrawEllipse(ell2[:,i], ax, rgb_r, alpha_value)

    for i in range(len(ell3[0,:])):
        DrawEllipse(ell3[:,i], ax, rgb_g, alpha_value)

    #ax.set_aspect('equal', 'datalim')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=2)

    DisplayPlot(fig,outfile)







def CreateModulePlot(array, title ="", ytitle="", labels=False, grid=False, outfile = False):    
    """
    Displays/Creates a plot with Mobius detector dimensions

    Parameters:        
        array:    1d np.array of the plot to be displayed
        title:    title for the plot
        ylabel:   title for the y-axis
        grid:     turns on grid lines
        outfile:  location to save the file
                
    Returns:
        Nothing
    """

    ys = np.array(array)

    dims = len(ys.shape)
    if (dims > 2):
       print("Dimenions of the array must be 1 or 2")
    
    if (dims==1):
        ys = np.reshape(ys, [1, len(ys)])

    print(ys.shape)
    if(ys.shape[1] != 31 and ys.shape[1] != 496):
        print("Warning!!! " + str(ys.shape[1]) + " elements in ModulePlot")
    
    x = np.linspace(0,ys.shape[1]-1,31)
    nPlots = ys.shape[0]
    if(labels==False):
        labels = np.repeat('',nPlots)
    
    fig = plt.figure()
    for i in range(nPlots):
        plt.plot(x,ys[i,:], label=labels[i])

    plt.title(title, fontsize=14)
    plt.ylabel(ytitle, fontsize=12)
    plt.xlabel('Detector Modules', fontsize=12)
    plt.xlim([0,30])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=10)

    if grid != False:
        ax.xaxis.grid(b=True, which='Major', linestyle='-', color='k')
        
    if outfile != False:
        plt.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        plt.close(fig)





def CreateDetectorImage(image, window=False, title ="", ctitle="",outfile = False):    
    """
    Displays/Creates a BW intensity image with Mobius detector dimensions

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
        
    if window == False:
        window = [np.min(image),np.max(image)]

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig = plt.figure()
    im = plt.imshow(image, vmin=window[0], vmax=window[1], cmap = cm.Greys_r)
    plt.title(title, fontsize=14)
    plt.ylabel('Module\nRows')
    plt.xlabel('Modules')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(16))
    ax.yaxis.set_major_locator(MultipleLocator(32))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticklabels(range(-1,33))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax,ticks=window)
    cbar.ax.set_ylabel(ctitle)
    cbar.ax.tick_params(labelsize=10) 

    if outfile != False:
        plt.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        plt.close(fig)

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'


def CreateDetectorMatrixImage(image, window=False, title ="", ctitle="",outfile = False):    
    """
    Displays/Creates a BW intensity image with Mobius detector dimensions

    Parameters:        
        image:    2d np.array of the image to be displayed
        window:   list or tuple of window range (min, max)
        title:    title for the image
        ctitle:   title for the colorbar
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
        
    if window == False:
        window = [np.min(image),np.max(image)]

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig = plt.figure()
    im = plt.imshow(image, vmin=window[0], vmax=window[1], cmap = cm.Greys_r)
    plt.title(title, fontsize=14)
    plt.ylabel('Modules')
    plt.xlabel('Modules')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticklabels(range(-1,33))
    ax.set_yticklabels(range(-1,33))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax,ticks=window)
    cbar.ax.set_ylabel(ctitle)
    cbar.ax.tick_params(labelsize=10) 

    if outfile != False:
        plt.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        plt.close(fig)

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'


def CreateDetectorHist(array, bins=500,title ="",xlabel="", outfile = False):    
    """
    Displays/Creates a histogram

    Parameters:        
        array:    n-d np.array of the values to be analyzed
        bins:     the number of bins for the histogram
        title:    title for the plot
        xlabel:   title for the c-axis
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    h = np.histogram(array,bins)

    fig = py.figure()
    ax = py.gca() 
    py.plot(h[1][1:],h[0])

    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    py.title(title)

    if outfile != False:
        py.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        py.close(fig)





def CreateAnnuliMasks(nPixels = 512, dPix = 1.0, center_pix=None):
    nDets = 496.
    nMods = 31


    if center_pix == None:
        center_pix = ()
        center_pix += (nPixels[0]/2,)
        center_pix += (nPixels[1]/2,)

    dDet = 512./nDets/dPix

    Dets = np.linspace(1,nDets/2,nDets/2)
    #Dets = np.concatenate(((Dets[::-1]),Dets))
    Mods = np.linspace(1,16,16)

    annuli_d = np.empty([nPixels,nPixels,len(Dets)])
    annuli_m = np.empty([nPixels,nPixels,len(Mods)])

    annuli_d[:,:,0] = pm.mask_circle(img_shape=(nPixels,nPixels), radius=dDet, center = center_pix)
    for d in Dets[:len(Dets)-1]: 
        annuli_d[:,:,d] = pm.mask_circle(img_shape=(nPixels,nPixels), radius=(d+1)*dDet, center = center_pix) - \
        pm.mask_circle(img_shape=(nPixels,nPixels), radius=d*dDet, center = center_pix)

    annuli_m[:,:,0] = pm.mask_circle(img_shape=(nPixels,nPixels), radius=8*dDet, center = center_pix)
    for m in Mods[:len(Mods)-1]:
        annuli_m[:,:,m] = pm.mask_circle(img_shape=(nPixels,nPixels), radius=(16*m+8)*dDet, center = center_pix) - \
        pm.mask_circle(img_shape=(nPixels,nPixels), radius=(16*(m-1)+8)*dDet, center = center_pix)

    return annuli_d, annuli_m


def CreateChannelPlot(array, x=False, title ="", ylabel="", grid=False, outfile = False):    
    """
    Displays/Creates a plot with Mobius detector dimensions

    Parameters:        
        array:    1d or 2d array
        title:    title for the plot
        ylabel:   title for the y-axis
        outfile:  location to save the file
                
    Returns:
        Nothing
    """


    ys = np.array(array)

    dims = len(ys.shape)
    if (dims > 2):
       print("Dimenions of the array must be 1 or 2")
    
    if (dims == 1):
        ys = reshape(ys, len(ys))

    nPlots = ys.shape[0]
       
    fig = plt.figure()
    for i in range(nPlots):
        py.plot(x,ys[i,:])

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Detector Modules', fontsize=12)
    plt.xlim([0,30])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=10)

    if grid != False:
        ax.xaxis.grid(b=True, which='Major', linestyle='-', color='k')
        
    if outfile != False:
        plt.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        plt.close(fig)

def CreateChannelMultiPlot(arrays, title ="",ylabel="", outfile = False):    
    """
    Displays/Creates a multiple line plot with Mobius detector dimensions

    Parameters:        
        array:    2d np.array of the plot to be displayed (nPlots,array)
        title:    title for the plot
        ylabel:   title for the y-axis
        outfile:  location to save the file
                
    Returns:
        Nothing
    """
    
    np.array(arrays)
    nPlots = arrays.shape[0]
    
    fig = plt.figure()
    for i in range(nPlots):
        py.plot(np.linspace(0,30,496),arrays[i,:])

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Detector Modules', fontsize=12)
    plt.xlim([0,30])

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=10)

    if grid != False:
        ax.xaxis.grid(b=True, which='Major', linestyle='-', color='k')
        
    if outfile != False:
        plt.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        plt.close(fig)    

   
      

def CreateDetectorMultHist(arrays, labels = False, bins=500,title ="",xlabel="", outfile=False):    
    """
    Displays/Creates a multiple histograms

    Parameters:        
        array:    n-d np.array of the values to be analyzed [nScans,data...]
        bins:     the number of bins for the histogram
        title:    title for the plot
        xlabel:   title for the c-axis
        outfile:  location to save the file
                
    Returns:
        Nothing
    """

    nScans = arrays.shape[0]
    if labels == False:
        labels = ["" for scan in range(nScans)]
    
    if len(labels) != nScans:
        print("Error! Lenths unequal")
        return
            
    fig = py.figure()
    ax = py.gca()
    for scan in range(nScans):
        h = np.histogram(arrays[scan,:], bins)
        py.plot(h[1][1:],h[0], label = labels[scan])

    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    ax.set_ylabel('Frequency')
    ax.set_xlabel(xlabel)
    py.title(title)
    
    if labels[0] != "":
        py.legend(loc=1, fontsize=12)

    if outfile != False:
        py.savefig(outfile + '.png', format = 'png', bbox_inches='tight')
        py.close(fig)
   

