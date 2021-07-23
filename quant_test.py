# -*- coding: utf-8 -*-

# quantitative evaluation

import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import trange
from PIL import Image #, ImageDraw
import math
import cv2

# find center of object with color rgb
# def find_pos(img, rgb, cnt_threshold, eps=40., show=0):
#     cnt = 0
#     x, y = [], []
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             flg = True
#             for k in range(3):
#                 if abs(img[i,j,k]-rgb[k]) > eps:
#                     flg = False
#                     break
#             if flg:
#                 x.append(float(j))
#                 y.append(float(i))
#                 cnt += 1
#     if cnt >= cnt_threshold and show:
#         cp = np.array(img)
#         for i,j in zip(x,y):
#             cp[int(j)][int(i)] = [255,255,255]
#         Image.fromarray(cp.astype(np.uint8)).show()
    
#     x_ = np.mean(x) if len(x)>0 else np.random.uniform(0,128)
#     y_ = np.mean(y) if len(y)>0 else np.random.uniform(0,128)
#     if cnt < cnt_threshold:
#         return -1., -1.
#     return x_, y_

def find_pos(img, rgb, cnt_threshold, show=0, seed=30):
    # img = (img.squeeze().detach().cpu().numpy() + 1) * 127.5
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'temp{seed}.png', img)
    img = cv2.imread(f'temp{seed}.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, rgb[0], rgb[1])
    cnt = mask.sum() / 255
    xy = np.argwhere(mask == 255)
    if cnt >= cnt_threshold and show:
        cp = np.array(img)
        name = np.random.randint(0, 1000000000)
        for i, j in xy:
            cp[i][j] = [255, 255, 255]
        Image.fromarray(cp.astype(np.uint8)).save(f'datagen/testing/{name}.png')
    if cnt < cnt_threshold:
        return -1., -1.
    x_ = xy.mean(0)[0]
    y_ = xy.mean(0)[1]
    return x_, y_


def test_roboarm(path, n_test, cnt_threshold, seed):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        color_dict = {'green': [(40, 50, 50), (70, 255, 255)],
            'yellow': [(25, 65, 65), (36, 255, 255)],
            'blue': [(110, 150, 150), (125, 255, 255)],
            'cyan': [(80, 150, 150), (95, 255, 255)],
            'red': [(0, 100, 100), (11, 255, 255)],
            'pink': [(138, 75, 75), (162, 255, 255)]
            }
        for c in color_dict.values():
            x1, y1 = find_pos(gt, np.array(c), cnt_threshold, seed=seed)
            if x1 < 0 or y1 < 0:
                continue
            x2, y2 = find_pos(pred, np.array(c), cnt_threshold, seed=seed)
            if x2 < 0 or y2 < 0:
                continue
            pos_eror = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            poserr.append(pos_eror)

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))


# def find_pos(img, rgb, cnt_threshold, eps=40., show=0):
#     cnt = 0
#     x, y = [], []
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             flg = True
#             for k in range(3):
#                 if abs(img[i,j,k]-rgb[k]) > eps:
#                     flg = False
#                     break
#             if flg:
#                 x.append(float(j))
#                 y.append(float(i))
#                 cnt += 1
#     if cnt >= cnt_threshold and show:
#         cp = np.array(img)
#         for i,j in zip(x,y):
#             cp[int(j)][int(i)] = [255,255,255]
#         Image.fromarray(cp.astype(np.uint8)).show()
    
#     x_ = np.mean(x) if len(x)>0 else np.random.uniform(0,128)
#     y_ = np.mean(y) if len(y)>0 else np.random.uniform(0,128)
#     if cnt < cnt_threshold:
#         return -1., -1.
#     return x_, y_

# for racket ball
def test_racket(path, n_test, cnt_threshold):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        x1, y1 = find_pos(gt, (255, 255, 255), cnt_threshold)
        x2, y2 = find_pos(pred, (255, 255, 255), cnt_threshold)
        poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))
        x1, y1 = find_pos(gt, (255, 0, 0), cnt_threshold)
        x2, y2 = find_pos(pred, (255, 0, 0), cnt_threshold)
        poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))

# for grid world
def test_shape(path, n_test, cnt_threshold):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        colors = [(255,0,0), (0,255,0), (255,0,255), (0,0,255), (255,255,0)]
        for c in colors:
            x1, y1 = find_pos(gt, c, cnt_threshold)
            if x1 < 0 or y1 < 0:
                continue
            x2, y2 = find_pos(pred, c, cnt_threshold)
            poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))

# for atari pong
def test_pong(path, n_test, cnt_threshold):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)[0:120]
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)[0:120]
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        colors = [(92,186,92), (213,130,74), (236,236,236)]
        for c in colors:
            x1, y1 = find_pos(gt, c, cnt_threshold)
            if x1 < 0 or y1 < 0:
                continue
            x2, y2 = find_pos(pred, c, cnt_threshold)
            poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))

'''
# for robot pushing
def test_push(path, n_test, cnt_threshold):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        colors = [(0, 255, 255), (229,229,20), (6,6,255), (5,255,5), (255,35,255)]
        for c in colors:
            x1, y1 = find_pos(gt, c, cnt_threshold)
            if x1 < 0 or y1 < 0:
                continue
            x2, y2 = find_pos(pred, c, cnt_threshold)
            if x2 < 0 or y2 < 0:
                continue
            poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))
'''


def test_push(path, n_test, cnt_threshold):
    l2diff, poserr = [], []
    for i in trange (n_test):
        pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)[15:128, 10:118]
        gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)[15:128, 10:118]
        l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
        l2diff.append(l2)
        
        colors = [[  0, 255,   0],
                [10, 10, 10],
                [235,   1, 235],
                [  0,   0, 255],
                [  0, 255, 255]]
        for c in colors:
            x1, y1 = find_pos(gt, c, cnt_threshold)
            if x1 < 0 or y1 < 0:
                continue
            x2, y2 = find_pos(pred, c, cnt_threshold)
            if x2 < 0 or y2 < 0:
                continue
            poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

    with open(os.path.join(path, 'acc.txt'), 'w') as f:
        f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
                np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))


# def test_roboarm(path, n_test, cnt_threshold):
#     l2diff, poserr = [], []
#     for i in trange (n_test):
#         pred = np.asarray(Image.open(os.path.join(path, '{}_pred.png'.format(i))), dtype=np.float32)
#         gt = np.asarray(Image.open(os.path.join(path, '{}_gt.png'.format(i))), dtype=np.float32)
#         l2 = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(gt))
#         l2diff.append(l2)
        
#         colors = [[  255, 26,   26],
#                 [255, 26, 255],
#                 [26,   255, 26],
#                 [  26,   26, 255],
#                 [  255, 255, 26],
#                 [ 26, 255, 255]]
#         for c in colors:
#             x1, y1 = find_pos(gt, c, cnt_threshold)
#             if x1 < 0 or y1 < 0:
#                 continue
#             x2, y2 = find_pos(pred, c, cnt_threshold)
#             if x2 < 0 or y2 < 0:
#                 continue
#             poserr.append(math.sqrt((x1-x2)**2+(y1-y2)**2))

#     with open(os.path.join(path, 'acc.txt'), 'w') as f:
#         f.write('mse: mean {} stddev {}; pos err.: mean {} stddev {}\n'.format(
#                 np.mean(l2diff), np.sqrt(np.var(l2diff)), np.mean(poserr), np.sqrt(np.var(poserr))))


# dispatcher
def test_acc(ds, path, n_test):
    if ds == 'racket':
        test_racket(path, n_test, 0)
    elif  ds == 'shape':
        test_shape(path, n_test, 100)
    elif ds == 'pong':
        test_pong(path, n_test, 1)
    elif ds == 'push':
        test_push(path, n_test, 10)
    elif ds == 'roboarm':
        seed = np.random.randint(0, 1000000000)
        test_roboarm(path, n_test, 10, seed)
    else:
        raise NotImplementedError
        