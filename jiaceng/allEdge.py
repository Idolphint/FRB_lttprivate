# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:30:56 2020

@author: litia
"""

from scipy.io import loadmat, savemat
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

path = 'E:/FRB/jiaceng/'
IMG=  'toGY20200319/Resize192ImgAug/'
LAB = 'toGY20200319/Resize192LabelAug/'
RES = 'toGY20200319/Res/'

EDGE3 = 'toGY20200319/edge3/' #轮廓



def EdgeBat(listN):
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        print(obj)
        getEdge(obj)