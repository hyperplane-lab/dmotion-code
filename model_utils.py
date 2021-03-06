import collections
import functools

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, affine_grid, grid_sample
import numpy as np

'''
# repara sampling
def gaussian_sampler(mean, log_var):
    x = torch.normal(torch.zeros(mean.size()), torch.ones(mean.size())).to(mean.device)
    return mean + x * torch.exp(log_var / 2.)
'''

'''
# channel wise conv
def conv_cross2d(inputs, weights, bias = None, stride = 1, padding = 0, dilation = 1, groups = 1):
    outputs = []
    for input, weight in zip(inputs, weights):
        output = conv2d(
            input = input.unsqueeze(0),
            weight = weight,
            bias = bias,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups
        )
        outputs.append(output)
    outputs = torch.cat(outputs, dim = 0)
    return outputs
'''

# shape: maps (batchsize, n_maps, size, size), thetas (batchsize, n_maps, 2, 3)
def spatial_transform2d(maps, thetas):
    n_maps = maps.shape[1]
    size = maps.shape[2]
    maps_ = maps.view(-1, 1, size, size)
    thetas_ = thetas.view(-1, 2, 3)
    grid = affine_grid(thetas_, maps_.size())
    maps_ = grid_sample(maps_, grid)
    return maps_.view(-1, n_maps, size, size)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, n_agent):
        super(GraphConvolutionLayer, self).__init__()
        self.attention = nn.Sequential(collections.OrderedDict([
            ('conv', Conv2dLayer(1, 128, 5, stride=2, normalization='batch', nonlinear='leakyrelu')),
            ('pool', nn.AdaptiveAvgPool2d(output_size=(1, 1))),
            ('flat', nn.Flatten(start_dim=1, end_dim=-1)),
        ]))

    @staticmethod
    def _get_adjacency(weights):
        # assume the graph is undirected, so the adj is symmetric
        weights = weights.transpose(2, 1)  # (B, 128, N)
        inner = -2 * torch.matmul(weights.transpose(2, 1), weights)
        x2 = torch.sum(weights**2, dim=1, keepdim=True)
        adj = x2 + inner + x2.transpose(2, 1)
        return adj

    @staticmethod
    def _normalize(adj, eps=1e-8):
        row_sum = torch.sum(adj, dim=1)
        row_inv = 1.0 / torch.sqrt(row_sum + eps)
        row_inv = torch.diag_embed(row_inv)
        adj = torch.bmm(torch.bmm(row_inv, adj), row_inv)
        return adj

    def forward(self, feature_maps):
        B, N, H, W = feature_maps.size()
        attention_features = feature_maps.reshape(B * N, H, W).unsqueeze(1)  # (B*N, 1, H, W)
        weights = self.attention(attention_features)
        weights = weights.reshape(B, N, -1)
        adj = self._normalize(self._get_adjacency(weights))  # (B, N, N)

        feature_maps = feature_maps.view(B, N, H * W)
        feature_maps = torch.bmm(adj, feature_maps)
        feature_maps = feature_maps.view(B, N, H, W)
        return feature_maps


class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 normalization='batch', nonlinear='relu'):
        if padding is None:
            padding = (kernel_size - 1) // 2

        bias = (normalization is None or normalization is False)

        modules = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            elif nonlinear == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(Conv2dLayer, self).__init__(*modules)

class DeConv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 normalization='batch', nonlinear='relu'):
        if padding is None:
            padding = (kernel_size - 1) // 2

        bias = (normalization is None or normalization is False)

        modules = [nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=stride-1,
            dilation=dilation,
            groups=groups,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            elif nonlinear == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(DeConv2dLayer, self).__init__(*modules)


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, normalization='batch', nonlinear='relu'):
        bias = (normalization is None or normalization is False)
        
        modules = [nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm1d(num_features=out_features))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(LinearLayer, self).__init__(*modules)

'''
class UBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc = None, submodule = None, outermost = False, innermost = False):
        super(UBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size = 4, stride = 2, padding = 1, bias = False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
'''

'''
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, ngf = 64):
        super(UNet, self).__init__()

        unet_block = UBlock(ngf * 8, ngf * 8, input_nc = None, submodule = None, innermost = True)

        for i in range(num_layers - 5):
            unet_block = UBlock(ngf * 8, ngf * 8, input_nc = None, submodule = unet_block)

        unet_block = UBlock(ngf * 4, ngf * 8, input_nc = None, submodule = unet_block)
        unet_block = UBlock(ngf * 2, ngf * 4, input_nc = None, submodule = unet_block)
        unet_block = UBlock(ngf, ngf * 2, input_nc = None, submodule = unet_block)
        unet_block = UBlock(out_channels, ngf, input_nc = in_channels, submodule = unet_block, outermost = True)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)
'''

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if m.weight is not None:
            nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, val = 0)

    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, mean = 1, std = 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, val = 0)


def init_weights(model):
    model.apply(functools.partial(_init_weights))