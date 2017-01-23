import os
import numpy as np
from CIFAR import CIFARPlotter
from Helpers import Helpers

########################################################################
# 2017 -  Manny Grewal
# This class helps with loading CIFAR resources. It reads pickled objects (binary objects) and returns their matrix equivalents
########################################################################

class CIFARLoader(object):
    # Width and height of each image.
    NUMBER_OF_PIXELS = 32
    # Number of channels in each image, 3 channels: Red, Green, Blue.
    TOTAL_CHANNELS = 3 
    # Number of example in one training set
    NUM_EXAMPLES = 10000     
    # Path of the CIFAR dataset ("Change this path to point to the location of your dataset")
    CIFAR_RESOURCES_PATH= r'D:\MP\DEV\Python\Resources\Datasets\cifar-10-python\cifar-10-batches-py\\'
    
    def __init__(self):
        pass

    def LoadFile(self, fileName):    
        batchFileName = os.path.join(self.CIFAR_RESOURCES_PATH,fileName)
        return Helpers.unpickle(batchFileName)
    
    # Convert the raw images from the data-files to floating-points. This method also returns the associated classNames matrix
    def GetFlattenedMatrix(self, fileName):    
        unpickledFile=self.LoadFile(fileName)
        # Get the raw images.
        rawImages = unpickledFile[b'data']
        # Get the class-numbers for each image. Convert to numpy-array.
        classNames = np.array(unpickledFile[b'labels'])
        # Convert the images.
        matrixImages = self.ConvertVectorToImage(rawImages)
        # Reshape 32 *32 * 3 (3D) vector into 3072 (1D) vector
        flattenedMatrix = np.reshape(matrixImages, (self.NUM_EXAMPLES, self.NUMBER_OF_PIXELS * self.NUMBER_OF_PIXELS * self.TOTAL_CHANNELS))
        
        return flattenedMatrix, classNames

    def ConvertVectorToImage(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0
        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self.TOTAL_CHANNELS, self.NUMBER_OF_PIXELS, self.NUMBER_OF_PIXELS])
        # Reorder the indices of the array. 10000x32x32x3
        images = images.transpose([0, 2, 3, 1])

        return images

    def LoadClassNames(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """

        # Load the class-names from the pickled file.
        rawLabels = self.LoadFile(fileName="batches.meta")[b'label_names']
        # Convert from binary strings.
        names = [x.decode('utf-8') for x in rawLabels]

        return names    




