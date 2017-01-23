import os
import numpy as np
import PlotCIFAR
from Helpers import Helpers
from kNN import kNN
%matplotlib inline

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
numberOfPixels = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
totalChannels = 3

# Size of test data
sizeTest = 10

# Path of the CIFAR dataset
cifarPath = r'D:\MP\DEV\Python\Resources\Datasets\cifar-10-python\cifar-10-batches-py\\'

def LoadFile(fileName):    
    batchFileName = os.path.join(cifarPath,fileName)
    return Helpers.unpickle(batchFileName)

def ConvertVectorToImage(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, totalChannels, numberOfPixels, numberOfPixels])

    # Reorder the indices of the array. 10000x32x32x3
    images = images.transpose([0, 2, 3, 1])

    return images

def LoadClassNames():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    rawLabels = LoadFile(fileName="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in rawLabels]

    return names    

data=LoadFile('data_batch_1')

# Get the raw images.
rawImages = data[b'data']

# Get the class-numbers for each image. Convert to numpy-array.
classNames = np.array(data[b'labels'])

# Convert the images.
matrixImages = ConvertVectorToImage(rawImages)

# Reshape 32 *32 * 3 (3D) vector into 3072 (1D) vector
matrixImages = np.reshape(matrixImages, (10000,3072))

names = LoadClassNames()

#Visualizing CIFAR 10
#arrayImages =[]
#arrayClassLabels=[]
#for j in range(6):
#    for k in range(6):
#        i = np.random.choice(range(len(matrixImages)))
#        arrayImages.append(matrixImages[i:i+1][0])
#        arrayClassLabels.append(names[classNames[i]])
        

#PlotCIFAR.PlotImages(arrayImages,arrayClassLabels,True)


nn = kNN.kNN() # create a Nearest Neighbor classifier class
nn.train(matrixImages, classNames) # train the classifier on the training images and labels
random_sample = np.random.randint(1, 9900)
Yte_predict,Ite_match = nn.predict(matrixImages[random_sample:random_sample+sizeTest]) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == classNames[random_sample:random_sample+sizeTest]) ))

print('Test data')
PlotCIFAR.PlotImages(matrixImages[random_sample:random_sample+sizeTest],classNames[random_sample:random_sample+sizeTest],True)
print('Predicted data')
PlotCIFAR.PlotImages(Ite_match[0:sizeTest],classNames[Yte_predict[0:sizeTest]],True)