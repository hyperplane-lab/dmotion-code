# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import interpolate
from model_utils import Conv2dLayer, LinearLayer, init_weights, spatial_transform2d, DeConv2dLayer, GraphConvolutionLayer



# image encoder: img(3, 128, 128)->maps(num_maps, map_size, map_size)
class ImageEncoder(nn.Module):
    def __init__(self, map_size, num_maps):
        super(ImageEncoder, self).__init__()

        self.map_size = map_size
        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(3, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(32, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv7', Conv2dLayer(32, num_maps, 5, stride=1, normalization=None, nonlinear=None)),
        ]))
    def forward(self, inputs):
        output = self.encoder.forward(interpolate(inputs, size=4*self.map_size, mode='nearest'))
        return output

'''
# VAE motion encoder: diff img(3, 128, 128)->mean(num_maps), var(num_maps)
class MotionEncoder(nn.Module):
    def __init__(self, num_maps=32):
        super(MotionEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(3, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(16, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(16, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(32, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(32, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, num_maps * 2, 5, stride=2, normalization=None, nonlinear=None)),
        ]))

    def forward(self, inputs):
        outputs = self.encoder.forward(inputs)
        outputs = outputs.view(inputs.size(0), -1)
        return torch.split(outputs, outputs.size(1) // 2, dim = 1)
'''

'''
# (frame1,frame2)(6, 128, 128) ->mean(num_maps*16), var(num_maps*16)
class MotionEncoder(nn.Module):
    def __init__(self, num_maps=32):
        super(MotionEncoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(6, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(16, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(16, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(32, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(32, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, num_maps * 2, 5, stride=1, normalization=None, nonlinear=None)),
        ]))

    def forward(self, f1, f2):
        inputs = torch.cat((f1, f2), 1)
        outputs = self.encoder.forward(inputs) # (batch, num_maps*2, 4, 4)
        outputs = outputs.view(inputs.size(0), -1)
        return torch.split(outputs, outputs.size(1) // 2, dim = 1)
'''

# deterministic encoder: (frame1,frame2)(6, 128, 128) -> z(num_maps*16)
class MotionEncoder(nn.Module):
    def __init__(self, num_maps=32, img_size=128):
        super(MotionEncoder, self).__init__()
        last_stride = 2 if img_size == 128 else 1
        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(6, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(16, 16, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(16, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(32, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(32, 64, 5, stride=last_stride, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv7', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv8', Conv2dLayer(64, num_maps, 5, stride=1, normalization=None, nonlinear=None)),
        ]))

    def forward(self, f1, f2):
        inputs = torch.cat((f1, f2), 1)
        outputs = self.encoder.forward(inputs) # (batch, num_maps, 4, 4)
        outputs = outputs.view(inputs.size(0), -1)
        return outputs

# transformation decoder: z(num_maps*16)->transform matrix(num_maps, 2, 3)
class TransformDecoder(nn.Module):
    def __init__(self, num_maps, translate_only):
        super(TransformDecoder, self).__init__()
        self.num_maps = num_maps
        self.decoders = nn.ModuleList()
        self.translate_only = translate_only
        for k in range(num_maps):
            self.decoders.append(
                nn.Sequential(
                    nn.Linear(16, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 2*3)
                )
            )
    
    # initialize the transformation to be identity
    def init_w(self):
        for k in range(self.num_maps):
            self.decoders[k][-1].weight.data.zero_()
            self.decoders[k][-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_maps, 16)

        outputs = []
        for i in range(self.num_maps):
            output = self.decoders[i].forward(inputs[:, i]) # (batch, 6)
            output = output.view(-1, 2, 3)
            outputs.append(output)
            #print(output.shape)

        outputs = torch.stack(outputs, dim = 1)
        outputs = outputs.view(-1, self.num_maps, 2, 3)
        if self.translate_only:
            outputs[:, :, 0, 0] = 1.
            outputs[:, :, 0, 1] = 0.
            outputs[:, :, 1, 0] = 0.
            outputs[:, :, 1, 1] = 1.
        #print(outputs)
        return outputs

# Image decoder: maps(num_maps, map_size, map_size)->img(3, 128, 128)
class ImageDecoder(nn.Module):
    def __init__(self, num_maps, img_size=128):
        super(ImageDecoder, self).__init__()
        
        self.decoder1 = nn.Sequential(OrderedDict([
                ('conv0', Conv2dLayer(num_maps, 32, 9, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv1', Conv2dLayer(32, 32, 9, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv2', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
            ]))
        self.decoder2 = nn.Sequential(OrderedDict([
                ('conv4', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv5', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv6', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv7', Conv2dLayer(32, 3, 5, stride=1, normalization=None, nonlinear='tanh')),
            ]))
        self.img_size = img_size
    
    def forward(self, inputs):
        if inputs.size(-1) < self.img_size:
            inputs = interpolate(inputs, size=self.img_size // 2, mode='nearest')
        up = self.decoder1.forward(inputs)
        up = interpolate(up, size=self.img_size, mode='nearest')
        return self.decoder2.forward(up)


# object extractor: frame1(3, 128, 128), frame2(3, 128, 128) -> frame2 pred(3, 128, 128)
# frame pixel range [-1,1]
# motion encoder is deterministic
class VDSTNModel(nn.Module):
    def __init__(self, num_maps, map_size, translate_only, img_size=128):
        super(VDSTNModel, self).__init__()
        
        # assert img_size == 128
        assert map_size == 32 or map_size == 64
        
        self.size = img_size
        self.map_size = map_size
        self.num_maps = num_maps

        self.image_encoder = ImageEncoder(num_maps=num_maps, map_size=map_size)
        self.motion_encoder = MotionEncoder(num_maps=num_maps, img_size=img_size)
        self.transform_decoder = TransformDecoder(num_maps=num_maps, translate_only=translate_only)
        self.image_decoder = ImageDecoder(num_maps = num_maps, img_size=img_size)

        init_weights(self)
        self.transform_decoder.init_w()
        

    def forward(self, frame1, frame2):
        z = self.motion_encoder.forward(frame1, frame2)
        features = self.image_encoder.forward(frame1)
        transform_mats = self.transform_decoder.forward(z)

        features_after = spatial_transform2d(features, transform_mats)

        # upsampling
        #if features_after.size(-1) != self.size:
        #    features_after = interpolate(features_after, size = self.size, mode = 'nearest')

        pred = self.image_decoder.forward(features_after)

        outputs = {
            'pred': pred,
            'features': features,
            'features_after': features_after,
            'transforms': transform_mats
        }
        return outputs

# interaction learner: frame0(3,128,128), frame1(3,128,128), map(n_agent,map_size,map_size) -> pred(3,128,128)
# support multi-agents
class G(nn.Module):
    def __init__(self, map_size, n_agent, img_size=128):
        super(G, self).__init__()
        
        assert map_size == 32 or map_size == 64
        # assert img_size == 128
        
        self.map_size = map_size
        self.img_size = img_size
        self.n_agent = n_agent
        
        self.down1 = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(6 + n_agent, 32, 5, stride=2, normalization='batch', nonlinear='leakyrelu'))
        ]))
        self.down2 = nn.Sequential(OrderedDict([
          ('conv1', Conv2dLayer(32 + n_agent, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(64, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(128, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(128, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(128, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu'))
        ]))
        self.up = nn.Sequential(OrderedDict([
          ('conv6', DeConv2dLayer(128 + n_agent, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv7', DeConv2dLayer(128, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv8', DeConv2dLayer(64, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv9', DeConv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv10', DeConv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv11', DeConv2dLayer(64, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv12', Conv2dLayer(32, 3, 3, stride=1, normalization=None, nonlinear='tanh'))
        ]))
    
        init_weights(self)

    def forward(self, input1, input2, mp):
        t = torch.cat((input1, input2, interpolate(mp, size=input1.size(-1), mode='nearest')), 1)
        t = self.down1.forward(t)
        t = torch.cat((t, interpolate(mp, size=t.size(-1), mode='nearest')), 1)
        t = self.down2.forward(t)
        t = torch.cat((t, interpolate(mp, size=t.size(-1), mode='nearest')), 1)
        t = self.up.forward(t)
        return t


# model
# train: triplet X(t-1),X(t),X(t+1) (3,128,128)*3 -> pred (3,128,128)
# interact: state X(t-1),X(t) (3,128,128)*2, action_mats (n_agent,2,3) -> pred (3,128,128) 
# use deterministic encoder, support multi-agent
class Model(nn.Module):
    def __init__(self, map_size, num_maps, n_agent, translate_only, args,
                 radius=0.07, h=0.03, eps=1e-12, img_size=128):
        super(Model, self).__init__()
        # assert img_size == 128
        assert map_size == 32 or map_size == 64

        self.use_gcn = (not args.no_graph) and (n_agent > 1)
        if self.use_gcn:
            self.gcn = GraphConvolutionLayer(n_agent)
        self.use_contrastive = args.use_contrastive
        self.use_landmark = args.use_landmark
        if self.use_landmark:
            self.landmark = PoseEncoder(num_maps, map_size)
            self.G_landmark = G(map_size, 1, img_size)

        self.n_agent = n_agent
        self.translate_only = translate_only

        self.G = G(map_size, n_agent, img_size)
        self.VD = VDSTNModel(num_maps, map_size, translate_only, img_size)

        self.radius = radius
        self.h = h
        self.eps=eps

    # def similarity_loss(self, feature_maps):
    #     B, C, H, W = feature_maps.size()
    #     feature_vectors = feature_maps.view(B, C, H * W)
    #     normalized_feature_vectors = F.normalize(feature_vectors, dim=-1)
    #     similarity_matrix = torch.bmm(
    #         normalized_feature_vectors,
    #         normalized_feature_vectors.detach().transpose(1, 2))
    #     triu_indices = similarity_matrix.triu(1).nonzero().T
    #     return similarity_matrix[triu_indices[0], triu_indices[1], triu_indices[2]].mean()

    def similarity_loss(self, feature_maps):
        B, C, H, W = feature_maps.size()
        feature_vectors = feature_maps.view(B, C, H * W)
        loss = 0.
        for c in range(C):
            other_indices = [i for i in range(C) if i != c]
            other_vectors = feature_vectors[:, other_indices, :]
            query_vector = feature_vectors[:, c, :].unsqueeze(1).repeat(1, C - 1, 1)
            difference = other_vectors - query_vector
            dist2 = torch.sum(difference ** 2, dim=1)
            dist2 = torch.max(dist2, torch.tensor(self.eps, device=dist2.device))
            dist = torch.sqrt(dist2)
            weight = torch.exp(- dist2 / self.h ** 2)
            loss += torch.mean((self.radius - dist) * weight)
        return loss

    @staticmethod
    def centroid(feature_map):
        x, _ = PoseEncoder.get_coord(feature_map, 2, feature_map.size(3))
        y, _ = PoseEncoder.get_coord(feature_map, 3, feature_map.size(2))
        return x, y

    def forward(self, frame0, frame1, frame2=None, action_mat=None):
        # train/test mode
        if frame2 is not None:
            bs = frame2.size(0)
            outputs_vd = self.VD.forward(frame1, frame2)
            maps_after = outputs_vd['features_after'] # (batch, num_maps, map_size, map_size)
            map_interactor = maps_after[:, 0:self.n_agent] # (batch, n_agent, map_size, map_size)
            if self.use_gcn:
                map_interactor = self.gcn(map_interactor)
            output_g = self.G.forward(frame0, frame1, map_interactor)
            if self.use_contrastive and self.n_agent > 1:
                contrastive_loss = self.similarity_loss(map_interactor)
            else:
                contrastive_loss = 0

            if self.use_landmark:
                _, gauss_xy = self.landmark(frame2)
                pred_g2 = self.G_landmark(frame0, frame1, gauss_xy)
                landmark_loss = F.mse_loss(pred_g2, frame2)
                gauss_x, gauss_y = self.centroid(gauss_xy)
                # TODO: setting the first map as gripper
                gripper_x, gripper_y = self.centroid(map_interactor[:, 0, :, :].unsqueeze(1))
                centroid_loss = torch.sqrt((gauss_x - gripper_x) ** 2 + (gauss_y - gripper_y) ** 2).mean()
            else:
                landmark_loss, centroid_loss = 0, 0
            return output_g, outputs_vd, contrastive_loss, landmark_loss, centroid_loss

        # interactive mode
        # action_mat (batch, n_agent, 2, 3)
        else:
            features = self.VD.image_encoder.forward(frame1) #(batch, num_maps, map_size, map_size)
            map_ = features[:,0:self.n_agent] #(batch, n_agent, map_size, map_size)
            map_after = spatial_transform2d(map_, action_mat) #(batch, n_agent, map_size, map_size)
            if self.use_gcn:
                map_after = self.gcn(map_after)  # TODO: before or after spatial transform?
            output_g = self.G.forward(frame0, frame1, map_after)
            #print(map_.size(), map_after.size(), output_g.size())
            return {'pred': output_g, 'map': map_, 'map_after': map_after}


class PoseEncoder(nn.Module):
    """Pose_Encoder:
    input: target image (transformed image)
    ouput: gaussian maps of landmarks
    https://github.com/hqng/imm-pytorch/imm_model.py
    """
    def __init__(self, num_maps, map_size, n_filters=1, gauss_std=0.1, gauss_mode='ankush'):
        super(PoseEncoder, self).__init__()
        self.map_size = map_size
        self.gauss_std = gauss_std
        self.gauss_mode = gauss_mode
        self.image_encoder = ImageEncoder(num_maps=num_maps, map_size=map_size)

        self.conv = self.conv_block(n_filters*8, 1, kernel_size=1, stride=1, batch_norm=False, activation=None)

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, dilation=1, bias=True, batch_norm=True, layer_norm=False, activation='ReLU'):
        padding = (dilation*(kernel_size-1)+2-stride)//2
        seq_modules = nn.Sequential()
        seq_modules.add_module('conv', \
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias))
        if batch_norm:
            seq_modules.add_module('norm', nn.BatchNorm2d(out_channels))
        elif layer_norm:
            seq_modules.add_module('norm', LayerNorm())
        if activation is not None:
            seq_modules.add_module('relu', getattr(nn, activation)(inplace=True))
        return seq_modules

    @staticmethod
    def get_coord(x, other_axis, axis_size):
        "get x-y coordinates"
        g_c_prob = torch.mean(x, dim=other_axis)  # B,NMAP,W
        g_c_prob = F.softmax(g_c_prob, dim=2) # B,NMAP,W
        coord_pt = torch.linspace(-1.0, 1.0, axis_size).to(x.device) # W
        coord_pt = coord_pt.view(1, 1, axis_size) # 1,1,W
        g_c = torch.sum(g_c_prob * coord_pt, dim=2) # B,NMAP
        return g_c, g_c_prob

    @staticmethod
    def get_gaussian_maps(mu, shape_hw, inv_std, mode='rot'):
        """
        Generates [B,NMAPS,SHAPE_H,SHAPE_W] tensor of 2D gaussians,
        given the gaussian centers: MU [B, NMAPS, 2] tensor.
        STD: is the fixed standard dev.
        """
        mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

        y = torch.linspace(-1.0, 1.0, shape_hw[0]).to(mu.device)

        x = torch.linspace(-1.0, 1.0, shape_hw[1]).to(mu.device)

        if mode in ['rot', 'flat']:
            mu_y, mu_x = torch.unsqueeze(mu_y, dim=-1), torch.unsqueeze(mu_x, dim=-1)

            y = y.view(1, 1, shape_hw[0], 1)
            x = x.view(1, 1, 1, shape_hw[1])

            g_y = (y - mu_y)**2
            g_x = (x - mu_x)**2
            dist = (g_y + g_x) * inv_std**2

            if mode == 'rot':
                g_yx = torch.exp(-dist)
            else:
                g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))

        elif mode == 'ankush':
            y = y.view(1, 1, shape_hw[0])
            x = x.view(1, 1, shape_hw[1])

            g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
            g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))

            g_y = torch.unsqueeze(g_y, dim=3)
            g_x = torch.unsqueeze(g_x, dim=2)
            g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

        else:
            raise ValueError('Unknown mode: ' + str(mode))

        return g_yx

    def forward(self, x_):
        x_ = self.image_encoder(x_)
        xshape = x_.shape
        x_ = self.conv(x_)

        gauss_y, gauss_y_prob = self.get_coord(x_, 3, xshape[2])  # B,NMAP
        gauss_x, gauss_x_prob = self.get_coord(x_, 2, xshape[3])  # B,NMAP
        gauss_mu = torch.stack([gauss_y, gauss_x], dim=2)

        gauss_xy = self.get_gaussian_maps(
            gauss_mu, [self.map_size, self.map_size], 1.0 / self.gauss_std, mode=self.gauss_mode)

        return gauss_mu, gauss_xy


'''
import numpy as np
enc = Model(map_size=64, n_agent=1, translate_only=False, num_maps=8)
device = torch.device('cpu')
enc = enc.to(device)
enc.eval()
d = torch.tensor(np.zeros((1, 3, 128, 128), dtype=np.float32))
m = torch.tensor(np.zeros((1,1,2,3), dtype=np.float32))
d = d.to(device)
enc(d, d, d)
'''
