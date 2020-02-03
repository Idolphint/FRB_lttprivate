# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:25:45 2020

@author: litia
"""

from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from torchvision import transforms, utils
from torch.utils.data.sampler import RandomSampler
import torch
import numpy as np
DEBUG=False
DATADIR = "E:/FRB/ALLdata/5Ddata/"
classNUM = 15
IMGLIST = "img"+str(classNUM)+"_list.txt"
LABELLIST="label"+str(classNUM)+"_list.txt"
class MyDataset(Dataset):
    def __init__(self, imgtxt, featxt, args, transform=None, target_transform=None,pin_memory=True):
        fh = open(imgtxt, 'r').readlines()
        fhfea = open(featxt, 'r').readlines()
        imgs = []
        if DEBUG:
            print("len img list:", len(fh))
            print("len label list: ",len(fhfea))
        for i in range(len(fh)):
            lineimg = fh[i].strip('\n')
            linefea = fhfea[i].strip('\n')
            imgs.append((lineimg, linefea))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.args = args
        
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = np.load(DATADIR+fn)
        label = np.load(DATADIR+label)
#        if classNUM ==2:
#            label2 = np.zeros((classNUM, self.args.base_size, self.args.base_size))
#            label2[0] = label
#            label2[1] = np.where(label < 1, 1, 0)
#            label = torch.from_numpy(label2)
#        if self.transform is not None:
#            img = self.transform(img)
#            label = self.transform(label).transpose(3,4)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        img = torch.unsqueeze(img, 0)
        label = torch.unsqueeze(label, 0).long()
        if DEBUG:
            print(img.size(), label.size())
            #label = torch.squeeze(label).long()
        return img,label
 
    def __len__(self):
        return len(self.imgs)

def make_data_loader(args, **kwargs):
    dataSet=MyDataset(imgtxt=DATADIR+IMGLIST,
                         featxt=DATADIR+LABELLIST,
                         args = args,
                         transform=transforms.ToTensor())
    validation_split = 0.1
    dataSetSize = len(dataSet)
    mid = int(dataSetSize*validation_split)
    val_indices, train_indices = list(range(0,mid)), list(range(mid,dataSetSize))
    train_loader = DataLoader(dataSet, batch_size=1, sampler=RandomSampler(train_indices))
    val_loader = DataLoader(dataSet, batch_size=1, sampler=RandomSampler(val_indices))
#    for i, obj in val_loader:
#        #a, b = obj['image'], obj['label']
#        print(i.shape)
#        print(obj.shape)
    return train_loader, val_loader, None, classNUM

if __name__ =="__main__":
    args=0
    
    make_data_loader(args)
    
