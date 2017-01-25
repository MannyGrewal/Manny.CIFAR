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

trainingDataSet, classesTraining= loader.GetFlattenedMatrix('data_batch_1')

testDataSet, classesTest= loader.GetFlattenedMatrix('test_batch')


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
nn.Train(trainingDataSet, classesTraining) # train the classifier on the training images and labels

random_sample = np.random.randint(1, 9900)
Yte_predict,Ite_match = nn.Predict(testDataSet[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH],True) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == classesTest[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH])))

print('Input data')
CIFARPlotter.PlotImages(testDataSet[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH],classesTest[random_sample:random_sample+WINDOW_SIZE_OF_TEST_BATCH],True)
print('Predicted data')
CIFARPlotter.PlotImages(Ite_match[0:WINDOW_SIZE_OF_TEST_BATCH],Yte_predict[0:WINDOW_SIZE_OF_TEST_BATCH],True)