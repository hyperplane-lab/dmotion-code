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
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tqdm import tqdm
from PIL import Image
from data import MyDataset
from tqdm import trange
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

# predict frame2 and visualize feature maps
def run_test(model, testset, args, device, epoch, mode='test'):
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
    def normalize(a):
        p = np.abs(a)
        mn, mx = np.min(p), np.max(p)
        return ((p - mn) / (mx - mn) * 255).astype(np.uint8)
    
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
    
    
    model.eval()
    #layout: im, extractor_pred, im_pred, feature map (t)
    #        diff, diff_extractor_pred, diff_pred, feature map (t+1)
    img_save = np.zeros([args.size * args.test_num * 2, args.size * 4, 3], dtype=np.uint8)
    for i in range(args.test_num):
        idx = random.randint(0, len(testset) - 1)
        sample, actions = testset.__getitem__(idx)
        frame0 = sample[0].to(device)
        frame1 = sample[1].to(device)
        frame2 = sample[2].to(device)
        
        diff_img = (sample[2] - sample[1]).numpy()
        img_save[args.size*(2*i+1): args.size*(2*i+2), 0: args.size, :] = diff2img(diff_img)
        img_save[args.size*(2*i): args.size*(2*i+1), 0: args.size, :] = frame2img(sample[2].numpy())
        
        pred_g, outputs_vd, _ = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), frame2.unsqueeze(0))
        pred_vd = outputs_vd['pred'].squeeze(0).cpu().detach().numpy()
        pred_g = pred_g.squeeze(0).cpu().detach().numpy()
        img_save[args.size*(2*i+1): args.size*(2*i+2), args.size: args.size*2, :] = \
                diff2img(pred_vd - sample[1].numpy(), need_check=False)
        img_save[args.size*(2*i+1): args.size*(2*i+2), args.size*2: args.size*3, :] = \
                diff2img(pred_g - sample[1].numpy(), need_check=False)
        img_save[args.size*(2*i): args.size*(2*i+1), args.size: args.size*2, :] = \
                frame2img(pred_vd, need_check=False)
        img_save[args.size*(2*i): args.size*(2*i+1), args.size*2: args.size*3, :] = \
                frame2img(pred_g, need_check=False)
        
        maps = outputs_vd['features'].squeeze(0).cpu().detach().numpy()
        maps_after = outputs_vd['features_after'].squeeze(0).cpu().detach().numpy()

        im = map2img(maps[0])
        img_save[args.size*(i*2): args.size*(i*2+1), args.size*(3): args.size*(4), :] = im
        im = map2img(maps_after[0])
        img_save[args.size*(i*2+1): args.size*(i*2+2), args.size*(3): args.size*(4), :] = im
        
    Image.fromarray(img_save).save(os.path.join(args.save_path, '{}_{}.jpg'.format(mode, epoch)))
    model.train()


def main(args):
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    plus = '_plus' if args.plus else ''
    contrastive = f'_contrastive_{args.contrastive_coeff}' if args.use_contrastive else ''
    agents = f'_agents_{args.n_agent}' if args.n_agent > 1 else ''
    graph = '_graph' if not args.no_graph else ''
    landmark = f'_landmark_{args.landmark_coeff}' if args.use_landmark else ''
    args.save_path = f'checkpoint_{args.size}{plus}{contrastive}{agents}{graph}{landmark}_{args.seed}'
    args.test_model_path = args.save_path + f'/model_{args.epochs - 1}.pth'
    args.test_save_path = f'test_{args.size}{plus}{contrastive}{agents}{graph}{landmark}_{args.seed}'

    if args.use_contrastive:
        assert args.n_agent > 1, 'Make sure the number of agent is more \
            than 1 when using contrastive loss'

    if args.plus:
        from models_plus import Model
    else:
        from models import Model
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    data, loaders = {}, {}
    data['train'] = MyDataset(
        data_path=args.data_path,
        mode='train',
        fmt = args.img_fmt,
        zoom = args.zoom,
        size=args.size
    )
    data['validate'] = MyDataset(
        data_path=args.data_path,
        mode='validate',
        fmt = args.img_fmt,
        zoom = args.zoom,
        size=args.size
    )
    data['test'] = MyDataset(
        data_path=args.data_path,
        mode='test',
        fmt = args.img_fmt,
        zoom = args.zoom,
        size=args.size
    )
    loaders['train'] = DataLoader(
        dataset=data['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    # print('dataset loaded, train {}, validate {}, test {}'.format(len(data['train']), len(data['validate']), len(data['test'])))
    model = Model(map_size=args.map_size, img_size=args.size, num_maps=args.num_maps, n_agent=args.n_agent,\
                  translate_only=args.translate_only, args=args).to(device)
    if args.deep_speed:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model,
            model_parameters=filter(lambda p: p.requires_grad, model.parameters()))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
    # print('start training ...')

    #run_test(model, data['test'], args, device, 0, 'test')
    # run_test(model, data['test'], args, device, 0, 'validate')
    writer = SummaryWriter(args.test_save_path + '/logs')

    for epoch in trange(args.epochs):
        n_iter = 0
        # print('start epoch {}'.format(epoch))
        train_losses = []
        losses_recon_vd, losses_recon_g, contrastive_losses = [], [], []
        centroid_losses, landmark_losses = [], []
        for batch, actions in loaders['train']:
            n_iter += 1
            if args.deep_speed:
                frame0, frame1, frame2 = batch[0].to(model_engine.local_rank), batch[1].to(
                    model_engine.local_rank), batch[2].to(model_engine.local_rank)
                pred_g, outputs_vd, contrastive_loss, landmark_loss, centroid_loss = model_engine(frame0, frame1, frame2)
            else:
                frame0 = batch[0].to(device)
                frame1 = batch[1].to(device)
                frame2 = batch[2].to(device)
                optimizer.zero_grad()
                pred_g, outputs_vd, contrastive_loss, landmark_loss, centroid_loss = model.forward(frame0, frame1, frame2)

            loss_recon_vd = F.mse_loss(outputs_vd['pred'], frame2)
            #loss_KL = -0.5 * torch.mean(1 + outputs_vd["logvar"] - outputs_vd["mean"].pow(2) - outputs_vd["logvar"].exp())
            loss_recon_g = F.mse_loss(pred_g, frame2)
            loss = (loss_recon_vd + loss_recon_g) * args.loss_scale
            loss += args.contrastive_coeff * contrastive_loss
            loss += centroid_loss
            loss += args.landmark_coeff * landmark_loss

            if args.deep_speed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            losses_recon_vd.append(loss_recon_vd.item())
            losses_recon_g.append(loss_recon_g.item())
            if args.use_contrastive and args.n_agent > 1:
                contrastive_losses.append(contrastive_loss.item())
            if args.use_landmark and args.n_agent > 1:
                centroid_losses.append(centroid_loss.item())
                landmark_losses.append(landmark_loss.item())
            # if n_iter % args.print_step == 0:
            #     print('epoch {}, step {}/{}, obj extractor loss: {}, interaction learner loss: {}, total loss: {}'.format(
            #             epoch, n_iter, len(data['train']) // args.batch, loss_recon_vd, loss_recon_g, loss))
        writer.add_scalar('Train Loss', sum(train_losses) / len(train_losses), global_step=epoch)
        writer.add_scalar('Train VD Loss', sum(losses_recon_vd) / len(losses_recon_vd), global_step=epoch)
        writer.add_scalar('Train G Loss', sum(losses_recon_g) / len(losses_recon_g), global_step=epoch)
        if args.use_contrastive:
            writer.add_scalar('Train Contrastive Loss', sum(contrastive_losses) / len(contrastive_losses), global_step=epoch)
        if args.use_landmark:
            writer.add_scalar('Train Centroid Loss', sum(centroid_losses) / len(centroid_losses), global_step=epoch)
            writer.add_scalar('Train Landmark Loss', sum(landmark_losses) / len(landmark_losses), global_step=epoch)
        if (epoch + 1) % args.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'model_{}.pth'.format(epoch)))
        # if (epoch + 1) % args.test_epoch == 0:
        #     #run_test(model, data['test'], args, device, epoch, 'test')
        #     run_test(model, data['test'], args, device, epoch, 'validate')
        if (epoch + 1) % args.validate_epoch == 0:
            loss1, loss2, loss3, loss4, loss5 = 0., 0., 0., 0., 0.
            model.eval()
            for i in range(data['validate'].__len__()):
                sample, actions = data['validate'].__getitem__(i)
                frame0 = sample[0].to(device)
                frame1 = sample[1].to(device)
                frame2 = sample[2].to(device)
                pred_g, outputs_vd, contrastive_loss, landmark_loss, centroid_loss = model.forward(frame0.unsqueeze(0), frame1.unsqueeze(0), frame2.unsqueeze(0))
                loss1 += F.mse_loss(outputs_vd['pred'], frame2).item()
                loss2 += F.mse_loss(pred_g, frame2).item()
                if args.use_contrastive:
                    loss3 += contrastive_loss.item()
                if args.use_landmark:
                    loss4 += landmark_loss.item()
                    loss5 += centroid_loss.item()
                del pred_g, outputs_vd, frame0, frame1, frame2
            model.train()
            # print('validate: epoch {}, obj extractor loss: {}, interaction learner loss: {}'.format(
            #         epoch, loss1, loss2))
            writer.add_scalar('Valid VD Loss', loss1 / data['validate'].__len__(), global_step=epoch)
            writer.add_scalar('Valid G Loss', loss2 / data['validate'].__len__(), global_step=epoch)
            if args.use_contrastive:
                writer.add_scalar('Valid Contrastive Loss', loss3 / data['validate'].__len__(), global_step=epoch)
            if args.use_landmark:
                writer.add_scalar('Valid Landmark Loss', loss4 / data['validate'].__len__(), global_step=epoch)
                writer.add_scalar('Valid Centroid Loss', loss5 / data['validate'].__len__(), global_step=epoch)


if __name__ == '__main__':
    from config import parser
    args = parser.parse_args()
    if args.deep_speed:
        import deepspeed

        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)

        args = parser.parse_args()
        args.deepspeeed_config = 'deepspeed_config.json'

    # print(args)

    main(args)
    