#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:37:18 2018

@author: lzj
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def plotFig(data,label,order):
    seq = [0,1,2,3,0]
    plt.figure()
    for i, j in enumerate(label):
        if j == 'Iris-setos':
            plt.plot(data[i,seq[order]],data[i,seq[order+1]],'ro')
        elif j == 'Iris-versicolo':
            plt.plot(data[i,seq[order]],data[i,seq[order+1]],'go')
        elif j == 'Iris-virginic':
            plt.plot(data[i,seq[order]],data[i,seq[order+1]],'bo')
        else:
            pass
    plt.show()
    
def readData(path):
    data = list()
    label = list()
    
    f = open(path,'r')
    l = f.readline()
    while (l != ''):
        l2 = list()
        a = l.split(',')
        for j in a[:-1]:
            l2.append(float(j))
        label.append(a[-1][0:-2])
        data.append(l2)
        l = f.readline()
    
    f.close()
    return np.array(data), np.array(label)

path = 'iris.data'
data, label = readData(path)


plotFig(data,label,0)
    
        
Len = len(label)
radio = 0.8
TrainLen = int(Len*radio)
seq = np.arange(Len)
np.random.shuffle(seq)
#random.sample(seq,150)

trainData = data[seq[:TrainLen]]
trainLabel = label[seq[:TrainLen]]
testData = data[seq[TrainLen:]]
testLabel = label[seq[TrainLen:]]

clf = svm.LinearSVC(loss='hinge')
clf.fit(trainData,trainLabel)

print("Predict...")

y_ = clf.predict(testData)
w = clf.coef_
b = clf.intercept_
print(w,b)
cnt = 0
for i,j in zip(testLabel,y_):
    if i == j:
        cnt += 1
acc = cnt * 1.0 /(Len-TrainLen)
print('The accuracy reuslt is: %.2f'%acc)
#print (clf.decision_function(testData))
