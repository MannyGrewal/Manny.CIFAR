import _pickle as cPickle
import os


def unpickle(filePath):
    with open(filePath, mode='rb') as fileObject:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        unpickledObject = cPickle.load(fileObject, encoding='bytes')
        fileObject.close()
    return unpickledObject
