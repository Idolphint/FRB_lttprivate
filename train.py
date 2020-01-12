from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from deeplab import DeepLab


DATADIR = "../data20200111/"

class MyDataset(Dataset):
    def __init__(self, imgtxt, labeltxt, transform=None, target_transform=None):
        fh = open(imgtxt, 'r').readlines()
        fhlabel = open(labeltxt, 'r').readlines()
        imgs = []
        for i in range(len(fh)):
            print(i)
            lineimg = fh[i].strip('\n')
            linelabel = fhlabel[i].strip('\n')
            imgs.append((lineimg, linelabel))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = loadmat(DATADIR+fn)['data']
        label = loadmat(DATADIR+label)['data']
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img,label
 
    def __len__(self):
        return len(self.imgs)
 
train_data=MyDataset(imgtxt='../data20200111/img_list.txt', labeltxt='../data20200111/label_list.txt', transform=transforms.ToTensor())
data_loader = DataLoader(train_data, batch_size=16,shuffle=True)
for epoch in range(10):
    print("epoch, ", epoch)
    for i, data in enumerate(data_loader):
        inputs, labels = data
        model = DeepLab(backbone='mobilenet', output_stride=16)
        model.eval()
        print(inputs.shape)
        output = model(inputs)
        print(output.size)
