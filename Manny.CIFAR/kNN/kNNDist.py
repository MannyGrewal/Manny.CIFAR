import numpy as np
import math

########################################################################
# 2017 -  Manny Grewal
# This class returns the L1 (Manhattan) and L2 (Euclidean) distance between two vectors i.e. test and X

def GetL2Distance(X, test):    
    return np.linalg.norm(test-X)

def GetL1Distance(X, test):    
    return np.sum(abs(test-X))