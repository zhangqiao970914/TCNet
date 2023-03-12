import torch
import time
from torch.optim import Adam
import network
from loss import structure_loss
import numpy as np
import cv2
from numpy import mean
from torch import nn
from dataset import get_loader
from os.path import join
import random
from utils import mkdir, write_doc, get_time


class Solver(object):
    def __init__(self):
        self.ICNet = network.ICNet().cuda()

    def train(self, roots, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path, num_thread, pin, vgg_path=None):
        # Define Adam optimizer.
        optimizer = Adam(self.ICNet.parameters(),
                         lr=learning_rate, 
                         weight_decay=weight_decay)

        # Load ".pth" to initialize model.
        if init_epoch == 0:
            # From pre-trained VGG16.
            self.ICNet.apply(network.weights_init)
            self.ICNet.vgg.vgg.load_state_dict(torch.load(vgg_path))
            #self.ICNet.rgb.load_state_dict(torch.load('/media/lab509-2/data/ZQ/T/ICNet/80.7_T2T_ViT_t_14.pth.tar'))
        else:
            # From the existed checkpoint file.
            ckpt = torch.load(join(ckpt_root, 'Weights_{}.pth'.format(init_epoch)))
            self.ICNet.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        # Define training dataloader.
        train_dataloader = get_loader(roots=roots,
                                      request=('img', 'gt'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        # Train.
        self.ICNet.train()
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0

            for data_batch in train_dataloader:
                self.ICNet.zero_grad()

                # Obtain a batch of data.
                img, gt = data_batch['img'], data_batch['gt']
                img, gt = img.cuda(), gt.cuda()

                if len(img) == 1:
                    # Skip this iteration when training batchsize is 1 due to Batch Normalization. 
                    continue
                
                # Forward.
                S_3_pred, S_4_pred, S_5_pred, S_g_pred = self.ICNet(image_group=img, is_training=True)


                # Compute IoU loss.
                loss1 = structure_loss(S_3_pred, gt)
                loss2 = structure_loss(S_4_pred, gt)
                loss3 = structure_loss(S_5_pred, gt)
                loss4 = structure_loss(S_g_pred, gt)
                loss = loss1 + loss3 + loss4 + loss4
                # Backward.
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.detach().item()
            
            # Save the checkpoint file (".pth") after each epoch.
            mkdir(ckpt_root)
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': self.ICNet.state_dict()}, join(ckpt_root, 'Weights_{}.pth'.format(epoch)))
            
            # Compute average loss over the training dataset approximately.
            loss_mean = loss_sum / len(train_dataloader)
            end_time = get_time()

            # Record training information (".txt").
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, learning_rate, end_time - start_time)
            write_doc(doc_path, content)
    
    def test(self, roots, ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # Load the specified checkpoint file(".pth").
            state_dict = torch.load(ckpt_path)['state_dict']
            self.ICNet.load_state_dict(state_dict)
            self.ICNet.eval()
            
            # Get names of the test datasets.
            datasets = roots.keys()

            # Test ICNet on each dataset.
            for dataset in datasets:
                # Define test dataloader for the current test dataset.
                test_dataloader = get_loader(roots=roots[dataset], 
                                             request=('img', 'file_name', 'group_name', 'size'),
                                             shuffle=False,
                                             data_aug=False, 
                                             num_thread=num_thread, 
                                             batch_size=batch_size, 
                                             pin=pin)

                # Create a folder for the current test dataset for saving predictions.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                for data_batch in test_dataloader:
                    # Obtain a batch of data.
                    img = data_batch['img'].cuda()




                    time_list = []
                    start_each = time.time()
                    # Forward.
                    preds = self.ICNet(image_group=img, 

                                       is_training=False)
                    time_each = time.time() - start_each
                    
                    print(time_each)
                    #print("{}'s average Time Is : {:.1f} fps".format(1 / mean(time_list)))
                    
                    # Create a folder for the current batch according to its "group_name" for saving predictions.
                    group_name = data_batch['group_name'][0]
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    # Make paths where predictions will be saved.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))
                    
                    # For each prediction:
                    for i, pred_path in enumerate(pred_paths):
                        # Resize the prediction to the original size when "original_size == True".
                        H, W = data_batch['size'][0][i], data_batch['size'][1][i]
                        pred = cv2.resize(preds[i], (W, H)) if original_size else preds[i]

                        # Save the prediction.
                        cv2.imwrite(pred_path, np.array(pred * 255))
