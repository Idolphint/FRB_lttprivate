# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:02:22 2020

@author: litia
"""

from scipy.io import loadmat, savemat
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.transform import radon, iradon


path = 'E:/FRB/jiaceng/'
#IMG=  'toGY20200319/Resize192ImgAug/'
#LAB = 'toGY20200319/Resize192LabelAug/'
IMG = 'hardImg/'
LAB = 'hardLab/'
#RES = 'toGY20200319/Res/'
EDGE1 = 'toGY20200319/edge1/'
EDGE2 = 'toGY20200319/edge2/'
EDGE3 = 'toGY20200319/edge3/'
RES = 'Res/ytyRes/'
VISUALIMG = 'toGY20200319/visualIMG/'
OUTCSV = "dicomJiaCeng.csv"
NAMELIST = "hardimg.txt"

def drawAttention(listN):
    for obj in open(os.path.join(path, listN)).readlines():
        img = obj.strip('\n')
        full_name = img.split('.')[0]
        ori_img = cv2.imread(os.path.join(path, 'toGY20200319/visualImg/', img))
        res = np.zeros_like(ori_img)
        #print(ori_img.shape,ori_img[0][0])
        lab = loadmat(os.path.join(path, LAB, full_name))['data']
        lab_edge = filters.sobel(lab)
        lab_edge = (lab_edge-  np.amin(lab_edge)) / (np.amax(lab_edge) - np.amin(lab_edge))*255
        lab_edge = lab_edge.astype(np.uint8)
        ret, lab_edge = cv2.threshold(lab_edge, 10, 255,cv2.THRESH_OTSU)
        res[lab_edge == 0] = ori_img[lab_edge == 0]
        res[lab_edge!=0] = 255
        print(res[lab_edge>0])
        cv2.imwrite(os.path.join(path, 'toGY20200319/LXYgif2/', img), res)
        

def betterEdge(bi_edge, ret, bi_edge_1):
    global_std = np.std(bi_edge)
    #print(ret, global_std)
    #########尝试将已经为1地区的影响去除
    po_1 = np.where(bi_edge > ret)
    
    bi_edge[po_1[0], po_1[1]] = 0
    bi_edge[po_1[0]+1, po_1[1]+1] = 0
    bi_edge[po_1[0]-1, po_1[1]+1] = 0
    bi_edge[po_1[0]+1, po_1[1]-1] = 0
    bi_edge[po_1[0], po_1[1]+1] = 0
    bi_edge[po_1[0]+1, po_1[1]] = 0
    bi_edge[po_1[0]-1, po_1[1]-1] = 0
    bi_edge[po_1[0]-1, po_1[1]] = 0
    bi_edge[po_1[0], po_1[1]-1] = 0
#        plt.imshow(bi_edge)
#        plt.show()
    
    #第一步， 全局阈值化==1 则==1， 小于0.3阈值也直接为0
    
    pos = np.where((bi_edge<ret) & (bi_edge>0.3*ret)) #&(bi_edge_1>0.12*ret)
    #pos = pos.transpose(1,0)
    #print(pos)
    pos = np.array(pos).transpose(1,0)
    #print(pos)
    #第二步，局部方差应该>0.15 的总方差，若不，则为0
    w,h = bi_edge.shape
    pad_edge = np.zeros((w+14, h+14))
    pad_edge[7:w+7,7:h+7] = bi_edge
    for xi, yi in pos:
        box = pad_edge[xi:xi+14, yi:yi+14]
        
#            buf = power
#            buf[xi-9+ari[0]:xi+5+ari[0], yi-9+ari[1]:yi+5+ari[1]] = 0
#            plt.imshow(buf)
#            plt.show()
        mean_box = np.mean(box)
        box_std = np.std(box)
        if box_std < 0.37*global_std:
            #print(xi," ", yi,"标准差太小啦", box_std)
            continue
        
        #第三部，局部阈值=E+2*方差
        if bi_edge[xi, yi] > mean_box+2*box_std:
            bi_edge_1[xi, yi] = 1
            #if (xi < 7 or yi < 7):
                #print("has small", box.shape)
            #print("add!!")
#        else :
#            print()#"局部阈值可能不对",mean_box,  box_std, bi_edge[xi, yi])  
    return bi_edge_1
    

def labelEdgeBat(listN):
#    f = open(os.path.join(path, OUTCSV), 'w', encoding='utf-8')
#    csv_writer = csv.writer(f)
#    csv_writer.writerow(["fileName", "hasJiaCeng"])
    
    for obj in open(os.path.join(path, listN)).readlines():
        obj = obj.strip('\n')
        print(obj)
        hasJiaCeng = labelEdge(obj)
        print(hasJiaCeng)
        #csv_writer.writerow([obj, hasJiaCeng])
        
        
def labelEdge(img):
    #%%加载图片
    full_name = img.split('.')[0]
    ori_img = loadmat(os.path.join(path, IMG, img), verify_compressed_data_integrity=False)['data']
    lab = loadmat(os.path.join(path, LAB, img))['data']
    #res = loadmat(os.path.join(path, RES, img))['data']
    
    lab = 1 - lab
    img = ori_img*lab
    #cv2.imwrite(os.path.join(path, EDGE2, full_name+'.jpg'), img)
#    plt.imshow(img)
#    plt.show()
    #%% 背景对比度降低
    
    lab_edge = filters.sobel(lab)
    lab_edge = (lab_edge-  np.amin(lab_edge)) / (np.amax(lab_edge) - np.amin(lab_edge))*255
    lab_edge = lab_edge.astype(np.uint8)
    ret, lab_edge = cv2.threshold(lab_edge, 10, 255,cv2.THRESH_OTSU)
    lab_edge = (lab_edge/255).astype(np.uint8)
    bg = np.mean(ori_img[lab_edge > 0]) 
#    plt.imshow(lab_edge)
#    plt.show()  #######输出label的边界
    
    img_adj = np.ones_like(img)*255
    img_adj[img>0] = img[img>0]
    img_adj[img <= 0] = bg
    img_adj = img_adj.astype(np.float32)
    #img_adj = sharpen(img_adj)
    plt.imshow(img_adj)
    plt.show()
    
    #%% 分区域
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
         
    #%%去噪、取轮廓、加粗
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
#    plt.imshow(ldimg)
#    plt.show()
    power = (ldimg - np.amin(ldimg))/ (np.amax(ldimg) - np.amin(ldimg)) *255
    power = power.astype(np.uint8) #0/1化了？？
    edge = np.zeros_like(power)
    edge2 = np.zeros_like(power)
    for ari in areas:
        buf = power[ari[0]-2:ari[2]+2, ari[1]-2:ari[3]+2]
#        ret, bi_power = cv2.threshold(buf, 0, 255,cv2.THRESH_OTSU)
#        kernel = np.ones((2, 2), np.uint8)
#        bi_power = cv2.erode(bi_power, kernel = kernel)
#        plt.imshow(bi_power)
#        plt.show()
        #all_edge = (all_edge/255).astype(np.uint8)
        edge_part = filters.sobel(buf)
        plt.imshow(edge_part)
        plt.show()
        
        #mag, ang = cv2.cartToPolar(gx, gy)
        bi_edge = (edge_part - np.amin(edge_part))/ (np.amax(edge_part) - np.amin(edge_part)) * 255
        bi_edge = bi_edge.astype(np.uint8)
        
        
        #%%如何二值化得到更好的结果呢？？
        #power = cv2.RemoveSmallRegion(power, 10, 1, 1)
        
        ret, bi_edge_1 = cv2.threshold(bi_edge, 0, 255,cv2.THRESH_OTSU)
        bi_edge_1 = betterEdge(bi_edge, ret, bi_edge_1)
        edge2[ari[0]-2:ari[2]+2, ari[1]-2:ari[3]+2] = bi_edge_1
        
        #%%
        edge[ari[0]-2:ari[2]+2, ari[1]-2:ari[3]+2] = bi_edge_1
        
    #kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    big_edge = cv2.dilate(edge, kernel = kernel)
    #big_edge = cv2.erode(big_edge, kernel = kernel)
#    plt.imshow(big_edge)
#    plt.show()
    #%% 着色
    #ldimg = img_adj

    line = np.where(edge2>0)
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

    colored = np.zeros_like(edge)
    #set 120=无夹层 240 = 有夹层
    for area in areas:
        buf = big_edge[area[0]-2:area[2]+2, area[1]-2:area[3]+2]
        w,h = buf.shape
        last_ed = 0
        padding=120
        if area[4] == 1:
            padding=240
        for xi in range(w):
            this_ed = 0
            line = np.zeros_like(buf[xi])
            po = np.where(buf[xi]>0)[0]
            if len(po)>0: #仅判断数组不为空
                lasti = po[0]
                for poi in po[1:]:
                    if poi-lasti >1:
                        line[lasti:poi] = padding
                        this_ed = poi
                    lasti = poi
                if (this_ed <last_ed - 12 or this_ed == 0) and last_ed !=0:
                    line[po[-1] : last_ed] = padding
                    this_ed = last_ed
                last_ed = this_ed
                buf[xi] = line
#        plt.imshow(buf)
#        plt.show()
#            cv2.floodFill(buf,mask, seedPoint=(c_x, c_y), newVal=120)
            #buf[buf>0] = 120
        colored[area[0]-2:area[2]+2, area[1]-2:area[3]+2] = buf
    plt.imshow(colored)
    plt.show()
    cv2.imwrite(os.path.join(path, RES, full_name+'.jpg'), colored)
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
    #drawAttention('toGY20200319/visualImg.txt')
#    G24_177
    #labelEdge("Group1_A_3_03110243-2_65.mat")
    #getSmoothEdge("Group1_A_10_3564401-2_301.mat", check=True)
    #SmoothEdgeBat("03img100.txt")