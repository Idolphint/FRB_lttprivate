import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm3d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab3d(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab3d, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm3d
        else:
            BatchNorm = nn.BatchNorm3d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True) #TODO

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm3d):
                m.eval()
            elif isinstance(m, nn.BatchNorm3d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv3d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                        or isinstance(m[1], nn.BatchNorm3d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv3d) or isinstance(m[1], SynchronizedBatchNorm3d) \
                        or isinstance(m[1], nn.BatchNorm3d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def getinput():
   root_path = "../data20200111/"
   img_path = "img/"
   fea_path = "feature/"
   label_path = "label/"




if __name__ == "__main__":
    model = DeepLab3d(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 1, 50, 513, 513) #B, C, D, W, H
    output = model(input)
    print(output.size())


