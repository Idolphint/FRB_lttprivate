from PIL import Image
from skimage import data, filters, img_as_ubyte, img_as_float
import matplotlib.pyplot as plt
import cv2
import os
from scipy.io import savemat
# 图像读取
pathfrom=r"C:\BUAA\FRB2\sobel\toGY20200319\out\anspic"
pathto=r"C:\BUAA\FRB2\sobel\toGY20200319\out\edge"

for image in range(39,167):
    img = plt.imread(os.path.join(pathfrom,str(image)+".jpg"))
    plt.imshow(img,plt.cm.gray)


    '''**********skimage*************'''
    # sobel边缘检测
    edges = filters.sobel(img)
    ymax = 255
    ymin = 0

    xmax = max(map(max,edges))
    xmin = min(map(min,edges))
    for i in range(len(edges)):
        for j in range(len(edges[0])):
            edges[i][j] = ((ymax-ymin)*(edges[i][j]-xmin)/(xmax-xmin))+ymin
    # 浮点型转成uint8型
    # 显示图像
    #plt.figure()
    #plt.imshow(edges,plt.cm.gray)

    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if(edges[i][j]>17):
                edges[i][j]=1
            else:
                edges[i][j]=0
    #im = Image.fromarray(edges)
    #im = im.convert("L")
    #im.save(os.path.join(pathto,str(image)+".jpg"))
    savemat(os.path.join(pathto,str(image)+".mat"),{'data':edges})