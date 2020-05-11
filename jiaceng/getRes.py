# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:08:50 2020

@author: litia
"""

from scipy.io import loadmat, savemat
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

path = 'E:/FRB/jiaceng/'
IMG=  'toGY20200319/Resize192ImgAug/'
LAB = 'toGY20200319/Resize192LabelAug/'
FEA = 'toGY20200319/myFea/'
RES = 'toGY20200319/Res/'

def visualFea(listN):
    for obj in open(os.path.join(path, listN)):
        name = obj.strip('\n')
        fea = loadmat(os.path.join(path, FEA, name))['data']
        print(fea.shape)
        res = np.argmax(fea, axis=0)
        savemat(os.path.join(path,RES, name), {'data': res})

    
if __name__ == "__main__":
    visualFea('03img.txt')