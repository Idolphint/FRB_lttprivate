
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
from deeplab3d import DeepLab3d
from utils.myDataSet import MyDataset,make_data_loader
import torch
#DATADIR = "../data20200111-img-label-fea/"
# 
#train_data=MyDataset(imgtxt='../data20200111-img-label-fea/img_list.txt', labeltxt='../data20200111-img-label-fea/label_list.txt', transform=transforms.ToTensor())
#data_loader = DataLoader(train_data, batch_size=3,shuffle=True)


import argparse
import os
import numpy as np
from tqdm import tqdm

from modeling.sync_batchnorm.replicate import patch_replication_callback

from deeplab3d import *

from utils.loss import DiceCELoss

from utils.calculate_weights import calculate_weigths_labels

from utils.lr_scheduler import LR_Scheduler

from utils.saver import Saver

from utils.summaries import TensorboardSummary

from utils.metrics import Evaluator


DEBUG=False
backbone  = 'mobilenet'
base_size = 384
EPOCH = 10
batch_size = 3
learning_rate = 0.0001
momentum = 0.9
weight_decay = 5e-4 #权重衰减率
ROOT_PATH = "E:/FRB"

CKPT_PATH = "ckpt"
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary

        # Define Dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # Define network

        model = DeepLab3d(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # Define Criterion
        # whether to use class balanced weights

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(ROOT_PATH, args.dataset+'_classes_weights.npy')

            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32)) ##########weight not cuda

        else:
            weight = None

        self.criterion = DiceCELoss()
        self.model, self.optimizer = model, optimizer

        

        # Define Evaluator

        self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler

        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,

                                            args.epochs, len(self.train_loader))



        # Using cuda

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):

        train_loss = 0.0
        dice_loss_count = 0.0
        ce_loss_count = 0.0
        num_count = 0

        self.model.train()

        tbar = tqdm(self.train_loader)

        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):

            image, target = sample
            if DEBUG:
                print("image, target size feed in model,", image.size(), target.size())
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(image)
            if DEBUG:
                print(output.size())
            n,c,d,w,h = output.shape
            output2 = torch.tensor( (np.zeros( (n,c,d,w,h) ) ).astype(np.float32) )
            if(output.is_cuda==True):
                output2 = output2.to(self.device)
            for mk1 in range(0,n):
                for mk2 in range(0,c): #对于每个n, c进行正则化
                    output2[mk1,mk2,:,:,:] = ( output[mk1,mk2,:,:,:] - torch.min(output[mk1,mk2,:,:,:]) ) / ( torch.max( output[mk1,mk2,:,:,:] ) - torch.min(output[mk1,mk2,:,:,:]) )
                
            loss, dice_loss, ce_loss = self.criterion(output,output2, target,self.device)

            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            dice_loss_count = dice_loss_count + dice_loss.item()
            ce_loss_count = ce_loss_count + ce_loss.item()
            num_count = num_count + 1

            # Show 10 * 3 inference results each epoch

            if i % (num_img_tr // 5) == 0:

                global_step = i + num_img_tr * epoch





        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        print('Loss: %.3f, dice loss: %.3f, ce loss: %.3f' % (train_loss, dice_loss_count/num_count, ce_loss_count/num_count))#maybe here is something wrong



        if self.args.no_val:

            # save checkpoint every epoch

            is_best = False

            self.saver.save_checkpoint({

                'epoch': epoch + 1,

                'state_dict': self.model.module.state_dict(),

                'optimizer': self.optimizer.state_dict(),

                'best_pred': self.best_pred,

            }, is_best)





    def validation(self, epoch):

        self.model.eval()

        self.evaluator.reset()

        tbar = tqdm(self.val_loader, desc='\r')

        test_loss = 0.0
        dice_loss = 0.0
        ce_loss = 0.0
        num_count = 0
        for i, sample in enumerate(tbar):
            image, target = sample
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)
            n,c,d,w,h = output.shape
            output2 = torch.tensor( (np.zeros( (n,c,d,w,h) ) ).astype(np.float32) )
            if(output.is_cuda==True):
                output2 = output2.to(self.device)
            for mk1 in range(0,n):
                for mk2 in range(0,c): #对于每个n, c进行正则化
                    output2[mk1,mk2,:,:,:] = ( output[mk1,mk2,:,:,:] - torch.min(output[mk1,mk2,:,:,:]) ) / ( torch.max( output[mk1,mk2,:,:,:] ) - torch.min(output[mk1,mk2,:,:,:]) )
                

            loss, dice, ce = self.criterion(output, ioutput2, target, self.device)
            test_loss += loss.item()
            dice_loss += dice.item()
            ce_loss += ce.item()
            num_count += 1
            tbar.set_description('Test loss: %.3f, dice loss: %.3f, ce loss: %.3f' % (test_loss / (i + 1), dice_loss / num_count, ce_loss / num_count))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
#            if self.args.cuda:
#                target, pred = torch.from_numpy(target).cuda(), torch.from_numpy(pred).cuda()
            self.evaluator.add_batch(np.squeeze(target), pred)



        # Fast test during the training

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        print('Loss: %.3f' % test_loss, dice_loss, ce_loss)



        new_pred = mIoU

        if new_pred > self.best_pred:

            is_best = True

            self.best_pred = new_pred

            self.saver.save_checkpoint({

                'epoch': epoch + 1,

                'state_dict': self.model.module.state_dict(),

                'optimizer': self.optimizer.state_dict(),

                'best_pred': self.best_pred,

            }, is_best)
            print("ltt save ckpt!")



def main():
    
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default=backbone,
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='heart_vessels',
                        choices=['pascal', 'coco', 'cityscapes', 'heart_vessels'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=base_size,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=base_size,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='diceCe',
                        choices=['ce', 'focal', 'log', 'dice', 'focal3d', 'ce3d', 'diceCe'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=EPOCH, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--train-from-begin', type=bool, default=True,
                        help='Set to load ckpt')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    if args.train_from_begin == False:
        ckpt_name = open(os.path.join(trainer.saver.directory,"best_ckpt_num.txt")).readline()
        trainer.model = trainer.model.load_state_dict(os.path.join(trainer.saver.directory, ckpt_name))
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        print("----------------------------epoch, ", epoch)
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)




if __name__ == "__main__":
   main()




#for epoch in range(10):
#    print("epoch, ", epoch)
#    for i, data in enumerate(data_loader):
#        inputs, labels = data
#        model = DeepLab(backbone='mobilenet', output_stride=16)
#        model.train()
#        print(inputs.shape)
#        output = model(inputs)
#        print(output.size)
