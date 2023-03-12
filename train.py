import os
from solver import Solver

vgg_path = '/user-data/T/ICNet/vgg16_feat.pth'
ckpt_root = './ckpt/'
train_init_epoch = 0
train_end_epoch = 20
train_device = '0'
train_doc_path = './training.txt'
learning_rate = 1e-5
weight_decay = 1e-4
train_batch_size = 10
train_num_thread = 4

# An example to build "train_roots".
train_roots = {'img': '/user-data/COCO+Class/images/',
               'gt': '/user-data/COCO+Class/gts/'}
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = train_device
    solver = Solver()
    solver.train(roots=train_roots,
                 vgg_path=vgg_path,
                 init_epoch=train_init_epoch,
                 end_epoch=train_end_epoch,
                 learning_rate=learning_rate,
                 batch_size=train_batch_size,
                 weight_decay=weight_decay,
                 ckpt_root=ckpt_root,
                 doc_path=train_doc_path,
                 num_thread=train_num_thread,
                 pin=False)

