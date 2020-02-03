import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
classNUM = 2
DEBUG=False
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 0.0001
    
        input_flat = input.view(N, -1)#压扁剩下全部
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        # loss = 1 - loss.sum() / N
        loss = 1 - loss
 
        return loss

class MulticlassDiceLoss2(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    # https://blog.csdn.net/CaiDaoqing/article/details/90457197

    def __init__(self):
        super(MulticlassDiceLoss2, self).__init__()
 
    def forward(self, input, target, weights=None):
        C = target.shape[0]
        dice = DiceLoss()
        totalLoss = 0
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            
            totalLoss += diceLoss
        return totalLoss/C
class MulticlassDiceLoss(nn.Module):

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
 
    def forward(self, input, target, weights=None): 

        C = target.shape[1]
        dice = DiceLoss()
        totalLoss = 0
 
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss/C


class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()
 
    def forward(self, input, target):
        n = target.size(0)
        h = target.size(2)
        w = target.size(3)

        smooth = 0.0001
        loss = 0
        # print(input.shape)
        # print(target.shape)

        # print( target.size(0),target.size(1),target.size(2),target.size(3) )
        d = torch.sub( input, target )
        d = torch.abs(d)
        loss = d.sum()
        loss = loss / (n*h*w)
            
        # print(d.shape)
        # print(loss)
        # sys.exit(1)
        return 2*loss


class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
 
    def forward(self, masks_pred, masks_pred2, true_masks, device):
        #require masks_pred原始预测结果
        #makss_pred2 对n,c分开正则化后的预测结果
        #true_masks 真实值

        n1,c1,d1,w1,h1 = masks_pred.shape
        SL_true_masks = SetMultiChannel(true_masks,device)
        # SL_true_masks = SL_true_masks.to(device)
        # masks_pred = NormalPre(masks_pred)

        dice_loss = MulticlassDiceLoss()
        # print( 'masks_pred', masks_pred.shape )
        # print( 'SL_true_masks', SL_true_masks.shape )
        dice_loss_num = 2 * dice_loss( masks_pred2 , SL_true_masks )
            
        masks_pred = torch.transpose( masks_pred,1,2)
        masks_pred = torch.transpose( masks_pred,2,3) #将类别放在最后 ncwh->nwhc // ncdwh -> ndwhc
        masks_pred = torch.transpose( masks_pred,3,4)
        masks_pred = torch.reshape(masks_pred, ( n1*d1*w1*h1 ,c1) )

        if DEBUG:
            print("before label view -1: ",true_masks.shape)
        true_masks = true_masks.permute( 0,2,3,4,1 )
        true_masks = true_masks.view(-1) #为什么true_masks不换维度呢？还是换一下好啦
        if DEBUG:
            print("after view -1: ", true_masks.shape)
        true_masks = true_masks.long()

        ce_loss = nn.CrossEntropyLoss()
        # ce_loss_num = 5 * ce_loss( masks_pred, true_masks )
        if DEBUG:
            print("pred size, gt size: ",masks_pred.size(), true_masks.size())
        ce_loss_num = 10 * ce_loss( masks_pred, true_masks )
        # 0-74 : 5 ; 74 - : 10 
        loss = dice_loss_num + ce_loss_num
        # print( 'Loss:', loss.item(), dice_loss_num.item(), ce_loss_num.item() )
        return loss, dice_loss_num, ce_loss_num

def NormalPre(data):
    n,c,w,h = data.shape
    for mk in range(0,n):
        for i in range(0,c):
            d1 = data[mk,i,:,:]
            d1_max = torch.max(d1)
            d1_min = torch.min(d1)
            data[ mk, i, :,: ] = ( d1 - d1_min ) / ( d1_max - d1_min )
    return data

# def SetMultiChannel(data):
#     n,w,h = data.shape
#     d = np.zeros( (n,6,w,h) )
#     for mk in range(0,n):
#         for i in range(0,w):
#             for j in range(0,h):
#                 d[ mk, int( data[ mk, i, j ] ), i, j ] = 1
#     return d
def SetMultiChannel(data,device): #one-hot deal
    n,c,d,w,h = data.shape
    emp = torch.tensor( np.zeros( (n,2,d,w,h) ).astype(np.float32) )
    emp = emp.to(device)
    for mk in range(0,n):
        emp[ mk, 0, :, :, : ] = torch.eq( data[mk,:,:,:], 0 )
        emp[ mk, 1, :, :, : ] = torch.eq( data[mk,:,:,:], 1 )
        #d[ mk, 2, :, : ] = torch.eq( data[mk,:,:], 2 )
        #d[ mk, 3, :, : ] = torch.eq( data[mk,:,:], 3 )
        #d[ mk, 4, :, : ] = torch.eq( data[mk,:,:], 4 )

    return emp





def build_loss(self, mode='ce3d'):
    """Choices: ['ce' or 'focal' , 'log'(cross) , 'dice', 'ce3d']"""
    if mode == 'ce':
        return self.CrossEntropyLoss
    elif mode == 'focal':
        return self.FocalLoss
    elif mode == 'log':
        return self.logLoss
    elif mode == 'ce3d':
        return self.CrossEntropyLoss3d
    elif mode == 'dice':
        return self.DiceLoss3d
    elif mode == 'diceCe':
        return DiceCeLoss
    else:
        raise NotImplementedError
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
    

    def to_one_hot(self, target, N):
        size =target.size()
        target = target.view(-1)
        
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, target)
        size.append(classNUM)
        return ones.view(*size)
    
    def to_one_hot3d(self, target, N):
        n,c,d,w,h = target.size()
        target_fla = target.flatten()
        size = target_fla.size()
        target_fla = target_fla.unsqueeze(1)

        #size1 = size.append(1)
        #sizeo_h = size.append(N)
        y_onehot = torch.FloatTensor(size,N)
        y_onehot.zero_()
        y_onehot.scatter_(1,target_fla,1)

        return y_onehot
    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        #target = torch.squeeze(target).long()
        
        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def DiceLoss3d(self, logit, target):
        n, c, d, h, w= logit.size()
        target = self.to_one_hot3d(target, classNUM)
        logit = logit.permute(0,2,3,4,1)
        target_f = torch.flatten(target)
        logit_f = torch.flatten(logit)
        intersection = torch.sum(target_f * logit_f) #and
        dice = (2. * intersection + smooth) / (torch.sum(target_f * target_f) + 
                torch.sum(logit_f * logit_f)+smooth)

        return (1. - dice) 
    def logLoss(self, logit, target):
        
        logit_softmax = F.softmax(logit)
        target_onehot = self.to_one_hot1(target, classNUM)
        
        if DEBUG:
            print("after onehot, target size: ", target_onehot.size())

    def CrossEntropyLoss3d(self, logit, target):
        
        label = self.tp_one_hot3d(target, N)
        logit = logit.permute(0,2,3,4,1)
        n,d,w,h,c = logit.size()
        logit = logit.view(n*d*w*h, c)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        #target = torch.squeeze(target).long()
        
        loss = criterion(logit, label)

        if self.batch_average:
            loss /= n
        return loss

    def FocalLoss3d(self, logit, target, gamma=2, alpha=0.5):
        n, c, d, h, w = logit.size()
        criterion = nn



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




