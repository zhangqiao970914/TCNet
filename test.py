import os
from solver import Solver


test_device = '0'
test_batch_size = 10
pred_root = './pred/'
ckpt_path = '/user-data/T/ICNet/ckpt/Weights_16.pth'
original_size = False
test_num_thread = 4

# An example to build "test_roots".
test_roots = dict()
datasets = ['CoSal2015','CoSOD3K','CoCA']

for dataset in datasets:
    roots = {'img': '/user-data/ICNet_Depth/Dataset/dataset_rgb/{}/images/'.format(dataset)}
    test_roots[dataset] = roots
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = test_device
    solver = Solver()
    solver.test(roots=test_roots,
                ckpt_path=ckpt_path,
                pred_root=pred_root, 
                num_thread=test_num_thread, 
                batch_size=test_batch_size, 
                original_size=original_size,
                pin=False)
