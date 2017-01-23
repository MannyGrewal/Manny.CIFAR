import matplotlib.pyplot as plt
import numpy as np
import math

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
    fig, axes1 = plt.subplots( totalRows, MAX_COLS, figsize=(5,5))   
    fig.tight_layout()

    arrayIndex=0
    for row in range(totalRows):
        for col in range(MAX_COLS):
            if(arrayIndex<totalImages):               
                axes1[row][col].set_axis_off()
                axes1[row][col].set_title(arrayClassLabels[arrayIndex])
                axes1[row][col].imshow(arrayImages[arrayIndex])
                arrayIndex+=1
