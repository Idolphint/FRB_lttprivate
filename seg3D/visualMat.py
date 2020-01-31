import os 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse

def readMatListandSave(path, name):
    for obj in open(path+name).readlines():
        obj = obj.strip("\n")
        mat = loadmat(path+obj)['data']
        plt.imshow(mat)
        plt.savefig(path+obj.split('.')[0]+".png")
        print("save ",path+obj.split('.')[0]+".png")
    print("finish")

def showPic(path, name):
    mat = loadmat(path+name)['data']
    plt.imshow(mat)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input mat name')   # 首先创建一个ArgumentParser对象
    parser.add_argument('path', type=str, help='please input mat or txt path')
    parser.add_argument('name', type=str, help='please input file name')

    args = parser.parse_args()
    path = args.path
    name = args.name
    if name.split('.')[-1] == "txt": #接受list文件可视化并保存
        readMatListandSave(path, name)
    else:
        showPic(path, name)
