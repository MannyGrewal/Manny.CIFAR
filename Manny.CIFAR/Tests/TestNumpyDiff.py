import numpy as np
import math


########################################################################
# 2017 -  Manny Grewal
# This is just a playground to learn & test various numpy functions

ints = np.random.randint(1, 20, size=30)
print(ints)
part = np.partition(ints,9)
print(part)
top2 = np.argpartition(ints, 9)[:9]
print(top2)
print(ints.take(top2))
print(np.bincount(ints.take(top2)))
print(np.bincount(ints.take(top2)).argmax())
#x = np.random.random_integers(1,10,size=(2,3,4))#y = np.random.random_integers(1,10,size=(2,1))#test = np.random.random_integers(1,10,size=(2,1))
#print("X \n")
#print(x.shape[0])
