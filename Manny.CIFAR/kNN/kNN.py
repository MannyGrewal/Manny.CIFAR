import numpy as np
import math

########################################################################
# 2017 -  Manny Grewal
# This class returns the k Nearest Neighbours of a vector

class kNN(object):
  def __init__(self):
    pass

  def Train(self, trainingData, trainingDataLabels):
    """ trainingData is stored in local variables """
    # trainingExample is like 50000 X 3072 array where each row is 3072 D vector of pixel values between 0 and 1
    self.trainingExamples = trainingData
    # labels is like 50000 X 3072 array where each row is the class value i.e. 0 to 9
    self.trainingLabels = trainingDataLabels

  def Predict(self, testData, predictedImages=False):
     # testData is the N X 3072 array where each row is 3072 D vector of pixel values between 0 and 1
    totalTestRows = testData.shape[0]
    # A vector where each element is zero with N rows where each row will be predicted class i.e. 0 to 9
    Ypred = np.zeros(totalTestRows, dtype = self.trainingLabels.dtype)
    Ipred = np.zeros_like(testData)

    # Iterate for each row in the test set
    for i in range(totalTestRows):
      # It uses Numpy broadcasting. Below is what is happening
      # testData[i,:] is test row of 3072 values
      # self.trainingExamples - testData[i,:] gives you a difference matrix of size 50000 X 3072 where each element is the difference value
      # np.sum() computes sums across the columns e.g. [ 2 4 9] sum is 15,
      # distances is 50000 rows where each element is the distance (cummulative sum of all 3072 columns) from test record (i)
      distances = np.sum(np.abs(self.trainingExamples - testData[i,:]), axis = 1)
      # get the index with smallest distance
      min_index = np.argmin(distances)
      # get the class value of the respective element from the labels array 
      Ypred[i] = self.trainingLabels[min_index] 
      # store the match
      Ipred[i] = self.trainingExamples[min_index] 
    return Ypred, Ipred