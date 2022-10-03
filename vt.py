import numpy as np
import matplotlib as mpl

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1 import make_axes_locatable

def round_sig(x, sig):
    """
    Rounds a number x to sig number of significant digits 

    Parameters
    ----------
    x : int or float
        Value to be rounded.
    sig : int
        The number of significant digits to round x.

    Returns
    -------
    float
        Rounded value.
    """
    
    #Condition to account for the log of 0
    if x == 0.0:
        return 0.0
    else:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def float2readablestr(x, sig=3):
    """
    Converts a float or int to a readable string value

    Parameters
    ----------
    x : int or float
        Value to be converted.
    sig : int
        The number of significant digits to round x. Default is 3

    Returns
    -------
    str
        A human readable string of x.
    """

    #Rounds the variable to sig number of signficant digits
    x = round_sig(x, sig)
    
    #If the value is within machine precission to 0 return '0'
    if np.isclose(x, 0):
        return "{:d}".format(int(np.rint(x)))

    return "{:g}".format(x)


#Sets the default font for all images
def FormatFig():
    mpl.rc('font', family='Times New Roman')


#Manipulates the y-axis to be the correct dimensions
def FormatYs(ys):
    ys = np.array(ys)

    nDims = ys.ndim
    if(nDims > 2):
       print("Dimenions of the array must be 1 or 2")
    
    if(nDims == 1):
        ys = ys[:,np.newaxis]

    return ys 


#Manipulates/creates the x-axis to be the correct dimensions
def FormatXs(xs,ys):
    nSamples, nPlots = ys.shape
    
    if xs is None:
        xs = np.linspace(0,nSamples-1,nSamples)
    else:
        xs = np.array(xs)

    nDims = xs.ndim
    if(nDims > 2):
        print("Dimenions of the array must be 1 or 2")

    xs = np.tile(xs, [nPlots,1]).T

    return xs


def FormatLabels(labels, nLabels):

    if labels is None:
        labels = np.repeat('',nLabels)
    
    if nLabels == 1:
        labels = [labels]

    return labels


def FormatLimits(ax,xlim,ylim):
    
    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    
    if ylim != None:
        ax.set_ylim(ylim[0],ylim[1])


def FormatTicks(ax,image,xlim,ylim,ticks):
    
    if xlim is not None:
        ticks = True
        ticks_loc = ax.get_xticks().tolist()
        tick_labels = np.array(ticks_loc)/(image.shape[1])* \
            (xlim[1] - xlim[0]) + xlim[0]
        
        """
        print(np.array(ticks_loc))
            
            
        print("HI")
        print(ax.get_xlim())
        k = ax.get_xlim()
        ax.set_xlim(xlim)
        ticks_loc = ax.get_xticks().tolist()
        ax.set_xlim(k)
        
        print(ax.get_xlim())
        
        print(np.array(ticks_loc))
        print(image.shape[1])
        print(xlim[1] - xlim[0])
        print(tick_labels)
        """
        
        tick_labels = [float2readablestr(x) for x in tick_labels]
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
        
    if ylim is not None:
        ticks = True
        ticks_loc = ax.get_yticks().tolist()
        tick_labels = np.array(ticks_loc)/(image.shape[0]-1)* \
            (ylim[1] - ylim[0]) + ylim[0]

        tick_labels = [float2readablestr(x) for x in tick_labels]
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))

    if ticks == False:
        RemoveTicks(ax)

def RemoveTicks(ax):
    ax.tick_params(length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')


def FormatColorBar(ax,im,vmin,vmax,ctitle):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks = [vmin,vmax])
    cbar.ax.set_ylabel(ctitle)
    cbar.ax.tick_params(labelsize=10) 


def DisplayPlot(fig, outfile):
    if outfile:
        plt.savefig(outfile + '.pdf',format='pdf',bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def LabelPlot(ax,title,xtitle,ytitle):
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(10)
        tick.set_fontname('Times New Roman')
        tick.set_color('black')

    ax.set_title(title, fontsize=14,fontname='Times New Roman')
    ax.set_xlabel(xtitle, fontsize=12,fontname='Times New Roman')
    ax.set_ylabel(ytitle, fontsize=12,fontname='Times New Roman')
    

def CreatePlot(ys,xs=None,err=None,title="",xtitle="",ytitle="",\
    xlims=None,ylims=None,scale=("Linear","Linear"),grid=False, \
    grid_minor=False, marker='',labels=None,outfile=False):    
    """
    Displays/Creates a plot

    Parameters:
        ys:       (array, nPlots)
        xs:       1d or 2d array
        title:    title for the plot
        ylabel:   title for the y-axis
        outfile:  location to save the file
        grid: False, x, y, both
                
    Returns:
        Nothing
    """
    
    ys = FormatYs(ys)
    xs = FormatXs(xs,ys)
    nSamples, nPlots = ys.shape
    labels = FormatLabels(labels, nPlots)

    fig, ax = plt.subplots()
    if err is None:
        for i in range(nPlots):
            ax.plot(xs[:,i],ys[:,i],label=labels[i],marker=marker)
    else:
        for i in range(nPlots):
            ax.errorbar(xs[:,i],ys[:,i],yerr=err,label=labels[i], fmt='o',capsize=3,capthick=.5, elinewidth=.5, ecolor='r', markersize=3)

    if labels[0] != '':
        ax.legend(loc=0, fontsize=10)


    ax.set_xscale(scale[0])
    ax.set_yscale(scale[1])
    
    if grid != False:
        if grid == "x":
            ax.grid(b=True, which='major', axis="x", linestyle='-', linewidth='0.25',color='k')
        elif grid == "y":
            ax.grid(b=True, which='major', axis="y", linestyle='-', linewidth='0.25',color='k')
        else:
            ax.grid(b=True, which='major', linestyle='-', linewidth='0.25',color='k')

    if grid_minor != False:
        if grid_minor == "x":
            ax.grid(b=True, which='minor', axis="x", linestyle='-', linewidth='0.25',color='k')
        elif grid_minor == "y":
            ax.grid(b=True, which='minor', axis="y", linestyle='-', linewidth='0.25',color='k')
        else:
            ax.grid(b=True, which='minor', linestyle='-', linewidth='0.25',color='k')


    FormatLimits(ax,xlims,ylims)
    LabelPlot(ax,title,xtitle,ytitle)
    DisplayPlot(fig,outfile)


def array2list(arr):
    l = []
    for i in range(arr.shape[1]):
        l.append(arr[:np.count_nonzero(arr[:,i]),i])

    return l


def CreateBoxPlot(samples,title="",xtitle="",ytitle="",labels="",outfile=False,remove_zeros=False,showfliers=True):
    '''
    samples (samples, expriments)
    
    '''

    #ys = FormatYs(ys)

    if remove_zeros == True:
        samples = array2list(samples)

    fig, ax = plt.subplots()
    ax.boxplot(samples,showfliers=showfliers)

    LabelPlot(ax,title,xtitle,ytitle)
    ax.set_xticklabels(labels)
    DisplayPlot(fig,outfile)


def CreateTiffImage(image, outfile=None):
    img = Image.fromarray(image.astype(np.float32))
    img.save(outfile+'.tif')


def CreateImage(image,window=False,title ="",xtitle="",ytitle="",ctitle="",\
                xvals=None,yvals=None,coords=None,outfile=None):
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
    
    if window == False:
        window = [np.min(image),np.max(image)]

    fig, ax = plt.subplots()
    #im = plt.imshow(image, vmin=window[0], vmax=window[1])
    im = plt.imshow(image, vmin=window[0], interpolation='None', vmax=window[1], cmap=cm.Greys_r, extent=coords, origin="lower")
    LabelPlot(ax,title,xtitle,ytitle)
    
    if coords == None:
        RemoveTicks(ax)
    

    if ctitle != "":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=window)
        cbar.ax.set_ylabel(ctitle)
        cbar.ax.tick_params(labelsize=10) 

    DisplayPlot(fig,outfile)


def imshow(image,vmin=None,vmax=None,title ="",xtitle="",ytitle="",ctitle="",\
           cmap=cm.Greys_r,ticks=False,xlim=None,ylim=None,extent=None,outfile=None):
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

    cmap = 'inferno'
    if vmin is None:
        vmin = np.min(image)

    if vmax is None:
        vmax = np.max(image)    

    fig, ax = plt.subplots()
    im = plt.imshow(image, vmin=vmin, vmax=vmax, interpolation='None', \
                    cmap=cmap, extent=extent, origin="lower")
    LabelPlot(ax,title,xtitle,ytitle)
    FormatTicks(ax,image,xlim,ylim,ticks)

    if ctitle != "":
        FormatColorBar(ax,im,vmin,vmax,ctitle)

    DisplayPlot(fig,outfile)


def imshow2(image,vmin=None,vmax=None,title ="",xtitle="",ytitle="",ctitle="",\
           cmap=cm.Greys_r,ticks=False,xlim=None,ylim=None,extent=None,outfile=None):
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

    cmap = 'inferno'
    if vmin is None:
        vmin = np.min(image)

    if vmax is None:
        vmax = np.max(image)    

    fig, ax = plt.subplots()
    im = plt.imshow(image, vmin=vmin, vmax=vmax, interpolation='None', \
                    cmap=cmap, extent=extent, origin="lower")
    #FormatTicks(ax,image,xlim,ylim,ticks)

    
    # Minor ticks
    #ax.set_xticks(np.arange(-.5, image.shape[1], 1), minor=True)
    #ax.set_yticks(np.arange(-.5, image.shape[0], 1), minor=True)
    
    # Gridlines based on minor ticks
    #ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    ax.xaxis.set_minor_locator(ticker.IndexLocator(base=1.0,offset=0.0))
    ax.yaxis.set_minor_locator(ticker.IndexLocator(base=1.0,offset=0.0))
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))

    ax.grid(which='minor', color='k', linestyle='-', linewidth=.5)
    #ax.grid(which='major', color='k', linestyle='-', linewidth=.5)

    #RemoveTicks(ax)
    DisplayPlot(fig,outfile)


def hist_show(arr, window=False, bins=None, title="", xtitle="", ytitle="",ctitle="",\
              xlabels=None, ylabels=None,outfile=False):


    if bins == None:
        binsX, binsY = arr.shape
    else:
        if arr.shape[0]%bins[0] != 0 or arr.shape[1]%bins[1] != 0:
            raise ValueError("Array must be evenly divisible by bins")
        else:
            binsX, binsY = bins

    if window == False:
        window = [np.min(arr),np.max(arr)]
            
    fig, ax = plt.subplots()
    im = plt.imshow(arr.T, origin='lower',cmap=cm.Greys_r)
    LabelPlot(ax,title,xtitle,ytitle)


    
    bin_samplesX = int(arr.shape[0]/binsX)
    bin_samplesY = int(arr.shape[1]/binsY)

    print(bin_samplesX)
    
    ax.set_xticks(np.linspace(-.5, binsX*bin_samplesX-.5, binsX+1))
    ax.set_xticklabels('')
    #ax.tick_params(length=0)
    
    if xlabels is not None:
        ax.set_xticks(np.linspace(bin_samplesX/2.0, bin_samplesX*(binsX - 0.5), binsX) - 0.5, minor=True)    
        ax.set_xticklabels(xlabels, minor=True)
    
    #ax.set_yticks(np.linspace(-.5, binsY*bin_samplesY-.5, binsY+1))
    ax.set_yticklabels('')
    if ylabels is not None:
        ax.set_yticks(np.linspace(bin_samplesY/2.0, bin_samplesY*(binsY - 0.5), binsY) - 0.5, minor=True)
        ax.set_yticklabels(ylabels,minor=True)
   
    ax.tick_params(axis='both', which='minor', length = 0.0)
    
    ax.grid(visible=True, which='Major', linestyle='-', linewidth='0.5',color='k')
    
    
    
    if ctitle != "":
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax, ticks=window)
        cbar.ax.set_ylabel(ctitle)
        cbar.ax.tick_params(labelsize=10) 
    
    
    DisplayPlot(fig,outfile)
