# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

#train
parser.add_argument('--dataset_name', default='push') # choices: 'shape', 'pong', 'push'
parser.add_argument('--gpu', default='0') # use gpu: 0, use cpu: 1
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--data_path', default='datagen/push') # dataset path
parser.add_argument('--img_fmt', default='.png') # .jpg or .png
parser.add_argument('--size', default=128, type=int)
parser.add_argument('--zoom', default=1.0, type=float)
parser.add_argument('--n_agent', default=1, type=int)
parser.add_argument('--num_maps', default=8, type=int) # number of feature maps
parser.add_argument('--map_size', default=64, type=int) # feature map size
parser.add_argument('--translate_only', default=True, type=bool) 
parser.add_argument('--plus', action='store_true', default=False)
parser.add_argument('--no_graph', action='store_true', default=False)
parser.add_argument('--contrastive_coeff', default=0.1, type=float)
parser.add_argument('--use_contrastive', action='store_true', default=False)
parser.add_argument('--use_landmark', default=False, action='store_true', help='use landmark generator')
parser.add_argument('--landmark_coeff', default=0.1, type=float)
# transformation matrix has 6 free parameters if translate_only is False; 
# otherwise it has 2 free parameters to model the translation only

parser.add_argument('--workers', default=4, type=int) # number of cpu workers

# Deep Speed
parser.add_argument('--deep_speed',
                    default=False,
                    action='store_true',
                    help='use DeepSpeed')
# cuda
parser.add_argument('--with_cuda',
                    default=False,
                    action='store_true',
                    help='use CPU in case there\'s no GPU support')
parser.add_argument('--use_ema',
                    default=False,
                    action='store_true',
                    help='whether use exponential moving average')

# train
parser.add_argument('-b',
                    '--batch_size',
                    default=32,
                    type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e',
                    '--epochs',
                    default=50,
                    type=int,
                    help='number of total epochs (default: 30)')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')

parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('--loss_scale', default=1.0, type=float)


parser.add_argument('--save_path', default='checkpoint') # path to save models and training results
parser.add_argument('--save_epoch', default=10, type=int) # every xx epoch to save model
parser.add_argument('--test_epoch', default=5, type=int) # every xx epoch to have a test
parser.add_argument('--validate_epoch', default=5, type=int) # every xx epoch to have a validate
parser.add_argument('--test_num', default=10, type=int)
parser.add_argument('--print_step', default=50, type=int) # every xx step to print loss
parser.add_argument('--idx_mpc', default=0, type=int)

#test
parser.add_argument('--test_model_path', default="checkpoint/model_49.pth")
parser.add_argument('--model_type', default="mood")
parser.add_argument('--test_save_path', default="test")
parser.add_argument('--visualize_feature', default=1, type=int) # test for visualize feature maps
parser.add_argument('--quantitative', default=1, type=int) # quantitative evaluation
parser.add_argument('--visual_forecasting', default=1, type=int) # test for visual forecasting conditioned on actions

# mpc
parser.add_argument('--l2', action='store_true', default=False)
parser.add_argument('--manhattan', action='store_true', default=False)
parser.add_argument('--mixed', action='store_true', default=False)
parser.add_argument('--sampled', action='store_true', default=False)
parser.add_argument('--num_iter', default=1, type=int)
parser.add_argument('--len_path', default=15, type=int)
parser.add_argument('--sum_tra', default=200, type=int)
parser.add_argument('--hide_penalty', default=False, type=float)
parser.add_argument('--total_steps', default=40, type=int)


parser.add_argument('--xml_path', default='/.../datagen/push_env.xml', type=str) # your absolute path of the robot xml file
