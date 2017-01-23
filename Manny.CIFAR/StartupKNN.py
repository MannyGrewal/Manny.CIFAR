import numpy as np
from kNN import kNN
from CIFAR import CIFARLoader
from CIFAR import CIFARPlotter
%matplotlib inline

########################################################################
# 2017 -  Manny Grewal
# This class is the entry point for the KNN classifier.
########################################################################

TRAINING_BATCH_FILENAME = 'data_batch_1'
TEST_BATCH_FILENAME = 'test_batch'
# Size of test data
WINDOW_SIZE_OF_TEST_BATCH = 10

loader = CIFARLoader.CIFARLoader()

data= loader.LoadFile('data_batch_1')

# Get the raw images.
rawImages = data[b'data']

# Get the class-numbers for each image. Convert to numpy-array.
classNames = np.array(data[b'labels'])

# Convert the images.
matrixImages = loader.ConvertVectorToImage(rawImages)

# Reshape 32 *32 * 3 (3D) vector into 3072 (1D) vector
matrixImages = np.reshape(matrixImages, (10000,3072))

names = loader.LoadClassNames()

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
Yte_predict,Ite_match = nn.predict(matrixImages[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH]) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == classNames[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH]) ))

print('Test data')
CIFARPlotter.PlotImages(matrixImages[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH],classNames[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH],True)
print('Predicted data')
CIFARPlotter.PlotImages(Ite_match[0:WINDOW_SIZE_OF_TEST_BATCH],classNames[Yte_predict[0:WINDOW_SIZE_OF_TEST_BATCH]],True)