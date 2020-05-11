# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:38:31 2020

@author: litia
"""

from scipy.io import loadmat, savemat
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.transform import radon, iradon
import csv


path = 'E:/FRB/jiaceng/'
IMG=  'toGY20200319/Resize192ImgAug/'
LAB = 'toGY20200319/Resize192LabelAug/'
#RES = 'toGY20200319/Res/'
EDGE1 = 'toGY20200319/edge1/'
EDGE2 = 'toGY20200319/edge2/'
EDGE3 = 'toGY20200319/edge3/'
RES = 'toGY20200319/visual-edge-res/'
VISUALIMG = 'toGY20200319/visualIMG/'
OUTCSV = "dicomJiaCeng.csv"
NAMELIST = "03img.txt"

def visual_mat(nameL):
    for obj in open(os.path.join(path, nameL)):
        obj = obj.strip('\n')
        obj = loadmat(os.path.join(path, IMG, obj))['data']
        label = loadmat(os.path.join(path, LAB, obj))['data']
        extract = np.multiply(obj,label)
        plt.imshow(extract)
        plt.show()
        break

def getSmoothEdge(img, check=False):
    #img = cv2.imread(os.path.join(path, IMG, img), flags=cv2.IMREAD_UNCHANGED)
    full_name = img.split('.')[0]
    ori_img = loadmat(os.path.join(path, IMG, img), verify_compressed_data_integrity=False)['data']
    lab = loadmat(os.path.join(path, LAB, img))['data']
    if check:
        plt.imshow(ori_img)
        plt.show()
    lab = 1 - lab
    img = ori_img*lab
    #####################提取对应的部分
    #plt.hist(img, bins=100)
    #plt.show()
    #aft = np.zeros_like(img)
    img_adj = np.ones_like(img)*255
    
    img_adj[img>50] = img[img>50]
    img_adj = img_adj.astype(np.float32)
    if check:
        plt.imshow(img_adj)
        plt.show()
    
    ldimg = cv2.bilateralFilter(img_adj, 4, 90, 90) #双边滤波过滤噪音，但是同时双边滤波会导致边缘不连续
    
    p_bg = np.sum(img)/np.sum(lab)
    ldimg[ldimg>p_bg] = p_bg   #消除背景带来的强对比影响

    print(np.amax(ldimg), np.amin(ldimg))
    if check:
        plt.imshow(ldimg)
        plt.show()
    power = (ldimg - np.amin(ldimg))/ (np.amax(ldimg) - np.amin(ldimg))
    power = power.astype(np.uint8)
    ldimg = filters.sobel(power)
    power = (ldimg - np.amin(ldimg))/ (np.amax(ldimg) - np.amin(ldimg)) * 255
    power = power.astype(np.uint8)
    
#    circle = cv2.HoughCircles(power, cv2.HOUGH_GRADIENT, #检测方法
#                              dp = 1.5,  #测圆心累加器图像的分辨率
#                              minDist = 10, #两圆心最小距离
#                                #传递给canny边缘算子的高阈值
#                              param2 = 60)  #圆心累加器阈值，越大越接近标准圆
    #cv2.imwrite(os.path.join(path, EDGE1, full_name+'.jpg'), power)
#    circle = np.squeeze(circle)
#    print(circle.shape)
#    #return
#    cv2.circle(power, (circle[0], circle[1]), circle[2], color=(126), thickness = 4)
    ret, mask = cv2.threshold(power, p_bg, 255,cv2.THRESH_OTSU)
    mask = (mask)/255
    img_edge = mask.astype(np.uint8)
    if check:
        plt.imshow(img_edge)
        plt.show()
        
        
        
    lab_edge = filters.sobel(lab)
    #print(np.amax(lab_edge), np.amin(lab_edge))
    lab_edge1 = (lab_edge - np.amin(lab_edge)) / (np.amax(lab_edge) - np.amin(lab_edge)) *255
    lab_edge1 = lab_edge1.astype(np.uint8)  
    
    ret,lab_edge = cv2.threshold(lab_edge1,50, 255,cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    lab_edge = cv2.dilate(lab_edge, kernel = kernel)
    lab_edge = lab_edge/255
    lab_edge = lab_edge.astype(np.uint8)
    plt.imshow(lab_edge)
    plt.show()
    print("label edge")
    res_mid = lab_edge - img_edge
    print(np.amax(res_mid), np.amin(res_mid))
    #res = np.clip(res_mid, 0, 1) * 255
    #cv2.imwrite(os.path.join(path, EDGE2, full_name+'.jpg'), res_mid)
    plt.imshow(res_mid)
    plt.show()
    #try1 = 
    #edges = filters.sobel(power)
    #edges = edges.astype(np.int)
#    print(np.amax(edges), np.amin(edges))
#    print(full_name)
    #cv2.imwrite(os.path.join(path, EDGE2, full_name+'.jpg'), edges)
#    edges_out = (edges - np.amin(edges)) / (np.amax(edges) - np.amin(edges)) *255.0
    #cv2.imwrite(os.path.join(path, EDGE2, full_name+'.jpg'), edges_out)
#    plt.imshow(edges_out)
#    plt.show()
    
    #edge = cv2.dilate()
#    
#    w,h = img.shape
#    mask = np.ones(img.shape, np.uint8)
#    mask[int(w/2-5): int(w/2+5), int(h/2-5):int(h/2+5)] = 0
#    f1 = np.fft.fft2(power)
#    f1shift = np.fft.fftshift(f1)
#    f1shift = f1shift*mask
#    f2shift = np.fft.ifftshift(f1shift)
#    aft = np.fft.ifft2(f2shift)
#    aft = np.abs(aft)
#    aft = (aft-np.amin(aft)) / (np.amax(aft) - np.amin(aft))
#    plt.imshow(aft)
#    plt.show()

def SmoothEdgeBat(listN):
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        print(obj)
        getSmoothEdge(obj)
        
def labelEdgeBat(listN):
    f = open(os.path.join(path, OUTCSV), 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["fileName", "hasJiaCeng"])
    
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        print(obj)
        hasJiaCeng = labelEdge(obj)
        print(hasJiaCeng)
        csv_writer.writerow([obj, hasJiaCeng])


def visualOriImg(listN):
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        full_name = obj.split('.')[0]
        print(obj)
        obj = loadmat(os.path.join(path, IMG, obj), verify_compressed_data_integrity=False)['data']
        obj = (obj - np.amin(obj) ) /(np.amax(obj) - np.amin(obj)) *255.
        cv2.imwrite(os.path.join(path, VISUALIMG, full_name+'.jpg'), obj)
#        cv2.imshow("ori", obj)
#        cv2.waitKey()
        

def sharpen(img):
    mean = np.mean(img)
    std = np.std(img)
    print(mean, mean-std, mean+std)
    res = np.zeros_like(img)
    res[img<=mean-std] = img[img<=mean-std] / (mean-std) *20
    res[img >= mean+std] = (img[img>=mean+std] - mean-std) / (255-mean-std) *20 + mean+std
    p = (img>mean-std) & (img<mean+std)
    res[p] =(img[p]-mean +std)/(2*std)*215 + mean-std
    return res

def directEdge(img):
    full_name = img.split('.')[0]
    ori_img = loadmat(os.path.join(path, IMG, img), verify_compressed_data_integrity=False)['data']
    lab = loadmat(os.path.join(path, LAB, img))['data']
    #res = loadmat(os.path.join(path, RES, img))['data']
    
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_img.jpg'), ori_img)
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_lab.jpg'), lab*255)
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_res.jpg'), res*255)
    lab = 1 - lab
    
    img = ori_img*lab

def labelEdge(img):
    full_name = img.split('.')[0]
    ori_img = loadmat(os.path.join(path, IMG, img), verify_compressed_data_integrity=False)['data']
    lab = loadmat(os.path.join(path, LAB, img))['data']
    #res = loadmat(os.path.join(path, RES, img))['data']
    
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_img.jpg'), ori_img)
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_lab.jpg'), lab*255)
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_res.jpg'), res*255)
    lab = 1 - lab
    
    img = ori_img*lab
    #cv2.imwrite(os.path.join(path, EDGE2, full_name+'.jpg'), img)
    plt.imshow(img)
    plt.show()
    #img_edge = filters.sobel(img)
    lab_edge = filters.sobel(lab)
    lab_edge = (lab_edge-  np.amin(lab_edge)) / (np.amax(lab_edge) - np.amin(lab_edge))*255
    lab_edge = lab_edge.astype(np.uint8)
    ret, lab_edge = cv2.threshold(lab_edge, 10, 255,cv2.THRESH_OTSU)
    lab_edge = (lab_edge/255).astype(np.uint8)
    
    bg = np.mean(ori_img[lab_edge > 0])
    lab_edge = 1-lab_edge
#    plt.imshow(lab_edge)
#    plt.show()  #######输出label的边界
    
    img_adj = np.ones_like(img)*255
    img_adj[img>0] = img[img>0]
    img_adj[img <= 0] = bg
    img_adj = img_adj.astype(np.float32)
    #img_adj = sharpen(img_adj)
    plt.imshow(img_adj)
    plt.show()
    print(np.amin(img_adj), bg)
    
#    cv2.imwrite(os.path.join(path, EDGE3, full_name+'_ext.jpg'), img_adj)
    ldimg = cv2.bilateralFilter(img_adj, 3, 90, 90) #双边滤波过滤噪音，但是同时双边滤波会导致边缘不连续
    
    #ldimg = cv2.medianBlur(img_adj, 5)
    #中值滤波后，强调夹层：夹层像素值-10
#    label_mean = np.mean(ldimg[lab>0])
#    po = (lab>0)&(ldimg<label_mean)
#    ldimg[po] = ldimg[po] - 10
    
#    plt.imshow(ldimg)
#    plt.show()
#    ldimg = sharpen(ldimg)
#    
    plt.imshow(ldimg)
    plt.show()
    #ldimg = img_adj
    lab = lab.astype(np.uint8)
    cts, hierarchy  = cv2.findContours(lab, mode=cv2.RETR_LIST, method=cv2.cv2.CHAIN_APPROX_NONE)
    
#    for ct in cts:
#        for pit in ct:
#            #print(pit[0], img_adj[pit[0][0], pit[0][1]])
#            img_adj[pit[0][1], pit[0][0]] = 255
    #cv2.drawContours(cv2.UMat(img_adj), cts, -1, 0, 3)
    
    areas = []
    for ct in cts:
        descri = []
        ct = np.squeeze(np.array(ct))
        if len(ct.shape) == 1:
            continue
        ct = ct.transpose(1,0)
        descri.append(np.min(ct[1]))
        descri.append(np.min(ct[0]))
        descri.append(np.max(ct[1]))
        descri.append(np.max(ct[0]))
        descri.append(0)   #记录是否有夹层
        areas.append(descri)
         
            
    
    power = (ldimg - np.amin(ldimg))/ (np.amax(ldimg) - np.amin(ldimg)) *255
    power = power.astype(np.uint8) #0/1化了？？
    ret, power = cv2.threshold(power, 10, 255,cv2.THRESH_OTSU)
    #all_edge = (all_edge/255).astype(np.uint8)
    ldimg = filters.sobel(power)
    
    #mag, ang = cv2.cartToPolar(gx, gy)
    power = (ldimg - np.amin(ldimg))/ (np.amax(ldimg) - np.amin(ldimg)) * 255
    power = power.astype(np.uint8)
    
    #power = cv2.RemoveSmallRegion(power, 10, 1, 1)
    ret, power = cv2.threshold(power, 10, 255,cv2.THRESH_OTSU)
    
    line = np.where(power>0)
    px = np.array(line[0], dtype=float)
    py = np.array(line[1], dtype=float)
    
    hasJiaCeng = False
    
    for area in areas:
        pxi = []
        pyi = []
#        power[area[0], area[1]]=255
#        power[area[0], area[3]]=255
#        power[area[2], area[1]]=255
#        power[area[2], area[3]]=255
        #power[int((area[0]+area[2]) / 2), int((area[1]+area[3])/2)] = 255
        for k in range(len(px)):
            if px[k] >area[0] and px[k] < area[2] and py[k]>area[1] and py[k] < area[3]:
                pxi.append(px[k])
                pyi.append(py[k])
        
        if (len(pxi) == 0):
            print("jump !!")
            continue
        mag, ang = cv2.cartToPolar(pxi - np.mean(pxi), pyi - np.mean(pyi))
#        plt.scatter(ang, mag)
#        plt.show()
        for ani in ang:
            
            po = (ang >= ani-0.1) & (ang <= ani+0.1)
            mai = mag[po]
            mai.sort()
            for t in range(len(mai) - 1):
                lenxd = mai[t+1] - mai[t]
                if (lenxd>3.5):
                    hasJiaCeng = True
                    break
            if hasJiaCeng:
                area[4] = 1
                hasJiaCeng = False
                break
    kernel = np.ones((2, 2), np.uint8)
    power = cv2.dilate(power, kernel = kernel)
    colored = np.zeros_like(power)
    #set 120=无夹层 240 = 有夹层
    for area in areas:
        buf = power[area[0]:area[2]+1, area[1]:area[3]+1]
        if area[4] == 1:
            print("yes")
            buf[buf>0] = 240
        else:
            buf[buf>0] = 120
        colored[area[0]:area[2]+1, area[1]:area[3]+1] = buf
    plt.imshow(colored)
    plt.show()
    #cv2.imwrite(os.path.join(path, RES, full_name+'.jpg'), colored)
    #print(hasJiaCeng)
    return hasJiaCeng
#            mai = mai[mai <= ani+0.2]
#            print(mai.shape)
#        for i in range(len(mag)):
#            for angi in angSet:
#                if ang[i] == 
        
    #cv2.imwrite(os.path.join(path, EDGE3, full_name+'.jpg'), power)
    #power[]
    
    
#    theta = np.linspace(0., 180., max(power.shape), endpoint=False)
#    sinogram = radon(power, theta=theta, circle=True)
#    plt.imshow(sinogram)
#    plt.show()
    
    
if __name__ == "__main__":
    #visual_mat("03img.txt")   只是提取后的区域哦
    #visualOriImg(NAMELIST)
    labelEdgeBat(NAMELIST)
    #labelEdge("Group1_A_2_0050163920-3_94.mat")
    #getSmoothEdge("Group1_A_10_3564401-2_301.mat", check=True)
    #SmoothEdgeBat("03img100.txt")