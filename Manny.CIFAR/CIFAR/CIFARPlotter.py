import matplotlib.pyplot as plt
import numpy as np
import math

########################################################################
# 2017 -  Manny Grewal
# Purpose of this class is to visualise a list of images from the CIFAR dataset

# Constants

# Width and height of each image.
maxImagesInARow = 6


#PlotImages method takes an list of Images and their respective labels in the second parameter
#Then it renders them using matplotlib imshow method in a 5 column matrix
def PlotImages(arrayImages,arrayClassLabels,reShapeRequired=False):
    totalImages=len(arrayImages)
    if(reShapeRequired==True):
        arrayImages = np.reshape(arrayImages, (totalImages,32,32,3))
    
    
    totalRows= math.ceil(totalImages/maxImagesInARow)
    fig, axes1 = plt.subplots(maxImagesInARow, totalRows, figsize=(6,6))
    
    arrayIndex=0
    for row in range(maxImagesInARow):
        for col in range(totalRows):
            if(arrayIndex<totalImages-1):               
                axes1[row][col].set_axis_off()
                axes1[row][col].set_title(arrayClassLabels[arrayIndex])
                axes1[row][col].imshow(arrayImages[arrayIndex])
                arrayIndex+=1
