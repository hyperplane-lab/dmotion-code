# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
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
          ('conv5', Conv2dLayer(64, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(32, num_maps, 5, stride=1, normalization=None, nonlinear=None)),
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
          ('conv5', Conv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', Conv2dLayer(64, num_maps, 5, stride=1, normalization=None, nonlinear=None)),
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
            ]))
        self.decoder2 = nn.Sequential(OrderedDict([
                ('conv2', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv3', Conv2dLayer(32, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
                ('conv4', Conv2dLayer(32, 3, 5, stride=1, normalization=None, nonlinear='tanh')),
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
          ('conv4', Conv2dLayer(128, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu'))
        ]))
        self.up = nn.Sequential(OrderedDict([
          ('conv5', DeConv2dLayer(128 + n_agent, 128, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv6', DeConv2dLayer(128, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv7', DeConv2dLayer(64, 64, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv8', DeConv2dLayer(64, 64, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv9', DeConv2dLayer(64, 32, 5, stride=1, normalization='batch', nonlinear='leakyrelu')),
          ('conv10', Conv2dLayer(32, 3, 3, stride=1, normalization=None, nonlinear='tanh'))
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
    def __init__(self, map_size, num_maps, n_agent, translate_only, args, img_size=128):
        super(Model, self).__init__()
        # assert img_size == 128
        assert map_size == 32 or map_size == 64
        
        self.n_agent = n_agent
        self.translate_only = translate_only
        
        self.G = G(map_size, n_agent, img_size)
        self.VD = VDSTNModel(num_maps, map_size, translate_only, img_size)
        self.use_gcn = (not args.no_graph) and (self.n_agent > 1)
        if self.use_gcn:
            self.gcn = GraphConvolutionLayer(n_agent)
        self.use_contrastive = args.use_contrastive

    def similarity_loss(self, feature_maps):
        B, C, H, W = feature_maps.size()
        feature_vectors = feature_maps.view(B, C, H * W)
        normalized_feature_vectors = F.normalize(feature_vectors, dim=-1)
        similarity_matrix = torch.bmm(
            normalized_feature_vectors,
            normalized_feature_vectors.detach().transpose(1, 2))
        triu_indices = similarity_matrix.triu(1).nonzero().T
        return similarity_matrix[triu_indices[0], triu_indices[1], triu_indices[2]].mean()

    def forward(self, frame0, frame1, frame2=None, action_mat=None):
        # train/test mode
        if frame2 is not None:
            outputs_vd = self.VD.forward(frame1, frame2)
            maps_after = outputs_vd['features_after'] # (batch, num_maps, map_size, map_size)
            map_interactor = maps_after[:, 0:self.n_agent] # (batch, n_agent, map_size, map_size)
            if self.use_gcn:
                map_interactor = self.gcn(map_interactor)
            output_g = self.G.forward(frame0, frame1, map_interactor)
            if self.use_contrastive:
                contrastive_loss = self.similarity_loss(map_interactor)
            else:
                contrastive_loss = 0
            return output_g, outputs_vd, contrastive_loss

        # interactive mode
        # action_mat (batch, n_agent, 2, 3)
        else:
            features = self.VD.image_encoder.forward(frame1) #(batch, num_maps, map_size, map_size)
            map_ = features[:,0:self.n_agent] #(batch, n_agent, map_size, map_size)
            if self.use_gcn:
                map_ = self.gcn(map_)  # TODO: before or after spatial transform?
            map_after = spatial_transform2d(map_, action_mat) #(batch, n_agent, map_size, map_size)
            output_g = self.G.forward(frame0, frame1, map_after)
            #print(map_.size(), map_after.size(), output_g.size())
            return {'pred': output_g, 'map': map_, 'map_after': map_after}
            

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


def main():
    from config import parser
    args = parser.parse_args()
    enc = Model(map_size=64, n_agent=6, translate_only=False, num_maps=8, args=args)
    d = torch.rand(1, 3, 128, 128)
    m = torch.rand(1, 1, 2, 3)
    output_g, outputs_vd, _ = enc(d, d, d)
    print(output_g.size())
    for k, v in outputs_vd.items():
        print(k, v.size())


if __name__ == '__main__':
    main()
