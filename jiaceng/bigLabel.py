# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:43:01 2020

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
BIGLABEL = 'toGY20200319/bigLabel/'
NAMELIST = "03img.txt"


def BigLabelBat(listN):
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        print(obj)
        BigLabel(obj)

def BigLabel(img):
    full_name = img.split('.')[0]
    #ori_img = loadmat(os.path.join(path, IMG, img), verify_compressed_data_integrity=False)['data']
    lab = loadmat(os.path.join(path, LAB, img))['data']
    res = np.ones((512,512))
    #lab = lab.resize((192*2, 192*2))
    buf = cv2.resize(lab, (192*2, 192*2))
    res[90:474, 72:456] = buf
    savemat(os.path.join(path, BIGLABEL, full_name+'.mat'),{'data': res})
    
    
if __name__ == "__main__":
    BigLabelBat(NAMELIST)