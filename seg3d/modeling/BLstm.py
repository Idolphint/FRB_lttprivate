import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init


class cutcube():
    def __init__(self, input_x, cubesize=7, stride=2):
        self.input_x = input_x
        self.cubesize = cubesize
        # self.padding=padding
        self.xdim = input_x.shape[0]
        self.ydim = input_x.shape[1]
        self.zdim = input_x.shape[2]
        self.stride = stride

    def cut(self):
        global nowx
        nowx=0
        global nowy
        global nowz
        add_z=torch.zeros(self.xdim,self.ydim,self.stride)
        add_y=torch.zeros(self.stride,self.ydim,self.zdim+self.stride)
        add_x=torch.zeros(self.xdim+self.stride,self.stride,self.zdim+self.stride)
        input_x=torch.cat((self.input_x,add_z),2)
        input_x=torch.cat((input_x,add_y),0)
        input_x=torch.cat((input_x,add_x),1)
        output = Variable()
        while nowx <= self.xdim:
            nowy = 0
            while nowy <= self.ydim:
                nowz = 0
                while nowz <= self.zdim:
                    temp = input_x[nowx:nowx + self.cubesize,nowy:nowy + self.cubesize,nowz:nowz + self.cubesize]
                    output = torch.cat((output, temp), 0)
                    nowz += self.stride
                nowy += self.stride
            nowx += self.stride
        return output


class cat:
    def __init__(self, input_x, input_y,input_z):
        self.input_x = input_x
        self.input_y = input_y
        self.input_z=input_z

    def catlayers(self):
        input_x = torch.reshape(self.input_x, (self.input_x.shape[0], -1))
        input_y = torch.reshape(self.input_y, (self.input_y.shape[0], -1))
        input_z = torch.reshape(self.input_z, (self.input_z.shape[0], -1))
        temp=torch.cat((input_x, input_y), 1)
        return torch.cat((temp, input_z), 1)


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=False, batch_first=True, dropout=0,
                 bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size  # 输入数据（一维向量）的元素个数，对于一维向量是他的长度
        self.hidden_size = hidden_size  # 隐藏层维度
        self.num_layers = num_layers  # 几个LSTM串联
        self.bias = bias
        self.batch_first = batch_first  # 默认开启，这样和我们的数据匹配
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first,
                              self.dropout, self.bidirectional)
        self.squeeze = nn.Linear(self.hidden_size*2, self.hidden_size)  #压缩两个biodirecctional方向
        self.squeeze2=nn.Linear(self.hidden_size,self.hidden_size//3)   #将三块串联的cube映射回原来的网络,并进行二分类

        print(self.bilstm)

    def forward(self, x):
        # 输入数据要求第一个维度是batchsize,第二个是序列的长度（分割的块数），第三个维度是数据的向量（一维）元素个数
        bilstm_out, _ = self.bilstm(x)  # 返回格式bilistm_out是batch,序列长度，隐藏层个数*输入数据的向量维数

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.squeeze(bilstm_out)
        y = self.squeeze2(y)
        logit = F.sigmoid(y)
        return logit

def bulid_bilstm(input_size, hidden_size, num_layers, bias=False, batch_first=True, dropout=0,
        bidirectional=True):
    return BiLSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional).cuda()
if __name__ == "__main__":
    testx = torch.rand((50, 50, 50))
    cutc = cutcube(testx, 7, 6)
    outx=cutc.cut()
    testground=torch.rand((50,50,50))
    cutc=cutcube(testground,7,6)
    outg=cutc.cut()

    testraw=torch.rand((50,50,50))
    cutc=cutcube(testraw,7,6)
    outraw=cutc.cut()
    catab = cat(outx, outg,outraw)
    outer = catab.catlayers()
    print(outer.shape)
    bilstm=BiLSTM(outer.shape[1],outer.shape[1],1)
    outer=outer.unsqueeze(0)
    answer=bilstm(outer)
    print(answer.shape)   #每个网络恢复原始大小







