import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab


########################################################################
# 2017 -  Manny Grewal
# Purpose of this class is to visualise a list of images from the CIFAR dataset

# How many columns to show in a grid
MAX_COLS = 5


#PlotImages method takes an list of Images and their respective labels in the second parameter
#Then it renders them using matplotlib imshow method in a 5 column matrix
def PlotImages(arrayImages,arrayClassLabels,reShapeRequired=False):
    totalImages=len(arrayImages)
    if(reShapeRequired==True):
        arrayImages = np.reshape(arrayImages, (totalImages,32,32,3))
    
    
    totalRows= math.ceil(totalImages/MAX_COLS)  
    fig = plt.figure(figsize=(5,5)) 
    gs = gridspec.GridSpec(totalImages, MAX_COLS)
    # set the space between subplots and the position of the subplots in the figure
    gs.update(wspace=0.1, hspace=0.4, left = 0.1, right = 0.7, bottom = 0.1, top = 0.9) 
    
    arrayIndex=0
    for g in gs:
        if(arrayIndex<totalImages):
            axes=plt.subplot(g)
            axes.set_axis_off()
            axes.set_title(arrayClassLabels[arrayIndex])
            axes.imshow(arrayImages[arrayIndex])
            arrayIndex+=1
    #plt.show()