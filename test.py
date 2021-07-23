# -*- coding: utf-8 -*-

#import argparse
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
from tqdm import trange
from PIL import Image
#from sklearn.decomposition import PCA

# from models import Model
from data import MyDataset
from quant_test import test_acc

###############################################################################
# utils
'''
def matvis_pca(act2mat, args, savefile='pca.png'):
    pca = PCA(n_components=2)
    data, label = [], []
    for k in act2mat.keys():
        for item in act2mat[k]:
            data.append(item.flatten().tolist())
            label.append(k)
    data_save = list(zip(data, label))
    data_save = np.array(data_save)
    np.save(os.path.join(args.test_save_path, 'pca.npy'), data_save, allow_pickle=True)
    
    d = pca.fit_transform(data)
    print('components ratio', pca.explained_variance_ratio_)
    
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'grey']
    act2color = {}
    for i, k in enumerate(act2mat.keys()):
        act2color[k] = colors[i]
    
    plt.close('all')
    ax = plt.subplot(111)
    for k in act2mat.keys():
        x, y = [], []
        for i in range(len(data)):
            if label[i] == k:
                x.append(d[i,0])
                y.append(d[i,1])
        ax.scatter(x, y, c=act2color[k], label=k)
    plt.legend()
    plt.savefig(os.path.join(args.test_save_path, savefile))
'''

def diff2img(diff, need_check=True): # [-2,2] (3,128,128) -> [0,255] (128,128,3)
    if need_check:
        assert np.max(diff) <= 2. and np.min(diff) >= -2.
    d = ((diff + 2.) * 127.5 / 2).astype(np.uint8)
    return np.transpose(d, (1,2,0))
def frame2img(fr, need_check=True): # [-1,1] (3,128,128) -> [0,255] (128,128,3)
    if need_check:
        assert np.max(fr) <= 1. and np.min(fr) >= -1.
    d = ((fr + 1.) * 127.5).astype(np.uint8)
    return np.transpose(d, (1,2,0))

'''
# visualize feature maps
def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2)
    return buf
def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())
def map2img(mp):
    plt.close('all')
    figure = plt.figure(figsize=(8,8))
    plot = figure.add_subplot(111)
    plot.axis('off')
    plot.matshow(mp)
    im = fig2img(figure).resize((args.size, args.size)).convert('L').convert('RGB')
    return np.asarray(im)[:,:,0:3]
'''

###############################################################################
# test functions
'''
def calc_act2mat(trainset, model, device):
    # visualize mat, compute centroids
    act2mats = [{} for j in range(args.n_agent)]
    # few-shot: first k samples in train set has label
    for i in trange(args.sample_num):
        idx = random.randint(0, len(trainset) - 1)
        sample, actions = trainset.__getitem__(idx)
        frame0 = sample[0].to(device)
        frame1 = sample[1].to(device)
        frame2 = sample[2].to(device)
        act = actions[1]
        
        pred_g, outputs_vd = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), frame2.unsqueeze(0))
        trans_mat = outputs_vd['transforms'][:, 0:args.n_agent].squeeze(0).cpu().detach().numpy() #(2,3)
        for j in range(args.n_agent):
            if act in act2mats[j].keys():
                act2mats[j][act].append(trans_mat[j])
            else:
                act2mats[j][act] = [trans_mat[j]]
    for j in range(args.n_agent):
        matvis_pca(act2mats[j], args, 'pca{}.png'.format(j))
    #print(act2mats)
    
    for j in range(args.n_agent):
        for k in act2mats[j].keys():
            m = np.asarray(act2mats[j][k])
            m = np.mean(m, axis=0)
            act2mats[j][k] = m #act2mats[j][k][0]

    return act2mats
'''

# find action-transformation table
def oneshot_act2mats(pairs, actions, model, device, args):
    act2mats = [{} for j in range(args.n_agent)]
    pth = os.path.join('datagen/demonstration', args.dataset_name)
    for p, a in zip(pairs, actions):
        frame0 = Image.open(os.path.join(pth, p[0]))
        frame0 = np.asarray(frame0.resize((args.size,args.size)), dtype=np.float32) / 127.5 - 1.0
        frame0 = torch.from_numpy(np.transpose(frame0, (2, 0, 1))).to(device).unsqueeze(0)
        frame1 = Image.open(os.path.join(pth, p[1]))
        frame1 = np.asarray(frame1.resize((args.size,args.size)), dtype=np.float32) / 127.5 - 1.0
        frame1 = torch.from_numpy(np.transpose(frame1, (2, 0, 1))).to(device).unsqueeze(0)
        
        out = model.VD.forward(frame0, frame1)
        trans_mat = out['transforms'][:, 0:args.n_agent].squeeze(0).cpu().detach().numpy()
        for j in range(args.n_agent):
            act2mats[j][a] = trans_mat[j]
    # put into same dict

    act2mats = {key: np.array([dict_[key] for dict_ in act2mats]) for key in act2mats[0].keys()}
    print(act2mats)
    return act2mats

# test in the whole testset, save ground truth and predicted x_{t+1}
def test_all(testset, model, act2mats, device, args):
    '''
    img_save = np.zeros([args.size * args.test_num, args.size * 7, 3], dtype=np.uint8)
    # test under interactive setting
    for i in range(args.test_num):
        idx = random.randint(0, len(testset) - 1)
        sample, actions = testset.__getitem__(idx)
        frame0 = sample[0].to(device)
        frame1 = sample[1].to(device)
        #frame2 = sample[2].to(device)
        act = actions[1]
        
        mat = []
        for j in range(args.n_agent):
            mat.append(act2mats[j][act])
        mat = np.asarray(mat)
        
        action_mat = torch.from_numpy(mat).to(device).unsqueeze(0)
        outputs = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), None, action_mat)
        pred = outputs['pred'].squeeze(0).cpu().detach().numpy()
        
        img_save[args.size*i: args.size*(i+1), 0: args.size, :] = frame2img(sample[1].numpy())
        img_save[args.size*i: args.size*(i+1), args.size: args.size*2, :] = frame2img(sample[2].numpy())
        img_save[args.size*i: args.size*(i+1), args.size*2: args.size*3, :] = frame2img(pred)
        img_save[args.size*i: args.size*(i+1), args.size*3: args.size*4, :] = diff2img(
                (sample[2]-sample[1]).numpy())
        img_save[args.size*i: args.size*(i+1), args.size*4: args.size*5, :] = diff2img(
                (pred-sample[1].numpy()))
        img_save[args.size*i: args.size*(i+1), args.size*5: args.size*6, :] = map2img(
                outputs['map'].squeeze(0).cpu().detach().numpy()[0])
        img_save[args.size*i: args.size*(i+1), args.size*6: args.size*7, :] = map2img(
                outputs['map_after'].squeeze(0).cpu().detach().numpy()[0])
    Image.fromarray(img_save).save(os.path.join(args.test_save_path, 'interact.png'))
    '''
    if not os.path.exists(os.path.join(args.test_save_path, 'img')):
        os.mkdir(os.path.join(args.test_save_path, 'img'))
    for i in trange(len(testset)):
        sample, actions = testset.__getitem__(i)
        frame0 = sample[0].to(device)
        frame1 = sample[1].to(device)
        #frame2 = sample[2].to(device)
        act = actions[1]
        action_mat = torch.from_numpy(act2mats[act]).to(device).unsqueeze(0).unsqueeze(0)
        outputs = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), None, action_mat)
        pred = outputs['pred'].squeeze(0).cpu().detach().numpy()
        
        Image.fromarray(frame2img(sample[1].numpy())).save(os.path.join(args.test_save_path, 'img', '{}_xt.png'.format(i)))
        Image.fromarray(frame2img(sample[2].numpy())).save(os.path.join(args.test_save_path, 'img', '{}_gt.png'.format(i)))
        Image.fromarray(frame2img(pred)).save(os.path.join(args.test_save_path, 'img', '{}_pred.png'.format(i)))

# visualize feature map of agent
def visfeaturemaps(testset, model, device, args, num=100):
    def normalize(a):
        p = np.abs(a)
        mn, mx = np.min(p), np.max(p)
        return ((p - mn) / (mx - mn) * 255).astype(np.uint8)
    
    if not os.path.exists(os.path.join(args.test_save_path, 'featuremap')):
        os.mkdir(os.path.join(args.test_save_path, 'featuremap'))
    for i in trange(num):
        sample, actions = testset.__getitem__(i)
        frame1 = sample[1].to(device)
        features = model.VD.image_encoder.forward(frame1.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        
        Image.fromarray(frame2img(sample[1].numpy())).save(os.path.join(args.test_save_path, \
                       'featuremap/{}_img.png'.format(i)))

        im = normalize(features[0])
        Image.fromarray(im).resize((args.size,args.size)).save(
                os.path.join(args.test_save_path, 'featuremap/{}_map.png'.format(i)))

# visual forecasting conditioned on actions
def longterm(video_id, model, act2mats, device, args, begin=0, maxlen=17, dr=None, act_seq=None):
    if not os.path.exists(os.path.join(args.test_save_path, 'video')):
        os.mkdir(os.path.join(args.test_save_path, 'video'))
    if not os.path.exists(os.path.join(args.test_save_path, 'video/{}'.format(video_id))):
        os.mkdir(os.path.join(args.test_save_path, 'video/{}'.format(video_id)))
    data_path = args.data_path if dr is None else dr
    print(data_path)
    
    with open(os.path.join(data_path, 'data.txt'), 'r') as f:
        lst, acts = [], []
        for l in f.readlines():
            nm = l.split()[0].split('_') # e.g. ['000001', '03']
            a = l.split()[1]
            lst.append(nm)
            acts.append(int(a))
    fn, act = [], []
    for f, a in zip(lst, acts):
        if f[0] == video_id and int(f[1]) >= begin:
            fn.append(os.path.join(data_path, f[0] + '_' + f[1] + args.img_fmt))
            act.append(a)

    gts = []
    for f in fn:
        frame0 = Image.open(f)
        frame0 = np.asarray(frame0.resize((args.size,args.size)), dtype=np.float32) / 127.5 - 1.0
        frame0 = torch.from_numpy(np.transpose(frame0, (2, 0, 1)))
        gts.append(frame0)
    frame0 = gts[0].to(device)
    frame1 = gts[1].to(device)
    if act_seq != None:
        act = act_seq
    for i in range(2, min(len(gts), maxlen)):
        action = act[i-1]
        
        # mat = []
        # for j in range(args.n_agent):
        #     mat.append(act2mats[action][j])
        # mat = np.asarray(mat)
        mat = act2mats[action]
        
        action_mat = torch.from_numpy(mat).to(device).unsqueeze(0)
        outputs = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), None, action_mat)
        pred = outputs['pred'].squeeze(0).cpu().detach().numpy()
        Image.fromarray(frame2img(pred)).save(os.path.join(args.test_save_path, 'video/{}/{}_{}.png'.format(
                video_id, begin + i, action)))
        frame0 = frame1
        frame1 = outputs['pred'].squeeze(0)


def main(args):
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    
    '''
    trainset = MyDataset(
        data_path=args.data_path,
        mode='train',
        fmt = args.img_fmt,
        size=args.size
    )
    '''
    
    testset = MyDataset(
        data_path=args.data_path,
        mode='test',
        fmt = args.img_fmt,
        size=args.size
    )

    print('dataset loaded, test {}'.format(len(testset)))

    model = Model(map_size=args.map_size, img_size=args.size, num_maps=args.num_maps, n_agent=args.n_agent,\
                  translate_only=args.translate_only, args=args).to(device)
    model.load_state_dict(torch.load(args.test_model_path, map_location=device))
    model.eval()
    
    if not os.path.exists(args.test_save_path):
        os.mkdir(args.test_save_path)
    
    # visualize feature maps
    if args.visualize_feature:
        visfeaturemaps(testset, model, device, args, num=100)
        
    # find action-transformation table via one demonstration
    pth = os.path.join('datagen/demonstration', args.dataset_name)
    with open(os.path.join(pth, 'demo.txt'), 'r') as f:
        img_pairs, actions = [], []
        for line in f.readlines():
            l = line.split()
            img_pairs.append((l[0], l[1]))
            actions.append(int(l[2]))
    print(img_pairs, actions)
    act2mats = oneshot_act2mats(img_pairs, actions, model, device, args)

    # visual forecasting conditioned on actions
    if args.visual_forecasting:
        i = 0
        while i < 10:
            idx = random.randint(0, 1000)
            idx = '%.6d' % (idx)
            try:
                longterm(idx, model, act2mats, device, args, begin=0)
                i += 1
            except:
                continue
    
    # quantitative test
    if args.quantitative:
        test_all(testset, model, act2mats, device, args)
        test_acc(args.dataset_name, os.path.join(args.test_save_path, 'img'), len(testset))
    

if __name__ == '__main__':
    from config import parser
    args = parser.parse_args()

    if args.plus:
        from models_plus import Model
    else:
        from models import Model

    plus = '_plus' if args.plus else ''
    contrastive = f'_contrastive_{args.contrastive_coeff}' if args.use_contrastive else ''
    agents = f'_agents_{args.n_agent}' if args.n_agent > 1 else ''
    graph = '_graph' if not args.no_graph else ''
    landmark = f'_landmark_{args.landmark_coeff}' if args.use_landmark else ''
    args.save_path = f'checkpoint_{args.size}{plus}{contrastive}{agents}{graph}{landmark}_{args.seed}'
    args.test_model_path = args.save_path + f'/model_{args.epochs - 1}.pth'
    args.test_save_path = f'test_{args.size}{plus}{contrastive}{agents}{graph}{landmark}_{args.seed}'

    main(args)
    