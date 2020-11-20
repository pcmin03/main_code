import cv2
import skimage
import torch
import numpy as np
import scipy
import math
from numpy.lib.stride_tricks import as_strided

from natsort import natsorted

import torch.nn.functional as F
from torch import nn

# natsort
from torch.nn.modules.utils import _pair, _quadruple
#=====================================================================#
#==============================++pooling=++===========================#
#=====================================================================#
class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=4, stride=1, padding=2, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

# Use with pytorch version >= 1.1.0

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

# print(torch.__version__)

from torch.nn.modules.utils import _pair, _quadruple
from torch.nn.modules.conv import _ConvNd
class GaborConv2d(_ConvNd):

    def __init__(self, in_channels=4, out_channels=4, kernel_size=50, stride=28,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',device='cpu'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.device = device

    def forward(self, input_image):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])])
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            weight = torch.empty(self.weight.shape, requires_grad=True).to(self.device)
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    sigma = self.sigma[i, j].expand_as(y)
                    freq = self.freq[i, j].expand_as(y)
                    theta = self.theta[i, j].expand_as(y)
                    psi = self.psi[i, j].expand_as(y)

                    rotx = x * torch.cos(theta) + y * torch.sin(theta)
                    roty = -x * torch.sin(theta) + y * torch.cos(theta)

                    g = torch.zeros(y.shape)

                    g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                    g = g * torch.cos(freq * rotx + psi)
                    g = g / (2 * 3.14 * sigma ** 2)
                    weight[i, j] = g
                    self.weight.data[i, j] = g
#         return self.weight,g,self.weight.data[i, j],self.freq
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

from scipy.ndimage import gaussian_filter
def g_difference(image, kernel, sigma1=3, sigma2=5):
    channels_in, channels_out = (image.shape[1], image.shape[1])
    diffs = gaussian_filter(image.cpu().detach().numpy(), sigma=sigma2) - gaussian_filter(image.cpu().detach().numpy() , sigma=sigma1)
    conv2d = nn.Conv2d(channels_in, channels_out, kernel_size=kernel, bias=False)
    with torch.no_grad():
        conv2d.weight = torch.nn.Parameter(torch.FloatTensor(diffs).cuda())
    return conv2d
# class GaborFilters(nn.Module):
#     def __init__(self, 
#         in_channels, 
#         n_sigmas = 3,
#         n_lambdas = 4,
#         n_gammas = 1,
#         n_thetas = 7,
#         kernel_radius=15,
#         rotation_invariant=True
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         kernel_size = kernel_radius*2 + 1
#         self.kernel_size = kernel_size
#         self.kernel_radius = kernel_radius
#         self.n_thetas = n_thetas
#         self.rotation_invariant = rotation_invariant
#         def make_param(in_channels, values, requires_grad=False, dtype=None):
#             if dtype is None:
#                 dtype = 'float32'
#             values = np.require(values, dtype=dtype)
#             n = in_channels * len(values)
#             data=torch.from_numpy(values).view(1,-1)
#             data = data.repeat(in_channels, 1)
#             # return data
#             return torch.nn.Parameter(data=data, requires_grad=False)


#         # build all learnable parameters
#         self.sigmas = make_param(in_channels, 2**np.arange(n_sigmas)*2)
#         self.lambdas = make_param(in_channels, 2**np.arange(n_lambdas)*4.0)
#         self.gammas = make_param(in_channels, np.ones(n_gammas)*0.5)
#         self.psis = make_param(in_channels, np.array([0, math.pi/2.0]))

#         print(len(self.sigmas))


#         thetas = np.linspace(0.0, 2.0*math.pi, num=n_thetas, endpoint=False)
#         thetas = torch.from_numpy(thetas).float()
#         self.register_buffer('thetas', thetas)

#         indices = torch.arange(kernel_size, dtype=torch.float32) -  (kernel_size - 1)/2
#         self.register_buffer('indices', indices)


#         # number of channels after the conv
#         self._n_channels_post_conv = self.in_channels * self.sigmas.shape[1] * \
#                                      self.lambdas.shape[1] * self.gammas.shape[1] * \
#                                      self.psis.shape[1] * self.thetas.shape[0] 


#     def make_gabor_filters(self):

#         sigmas=self.sigmas
#         lambdas=self.lambdas
#         gammas=self.gammas
#         psis=self.psis
#         thetas=self.thetas
#         y=self.indices
#         x=self.indices

#         in_channels = sigmas.shape[0]
#         assert in_channels == lambdas.shape[0]
#         assert in_channels == gammas.shape[0]

#         kernel_size = y.shape[0], x.shape[0]



#         sigmas  = sigmas.view (in_channels, sigmas.shape[1],1, 1, 1, 1, 1, 1)
#         lambdas = lambdas.view(in_channels, 1, lambdas.shape[1],1, 1, 1, 1, 1)
#         gammas  = gammas.view (in_channels, 1, 1, gammas.shape[1], 1, 1, 1, 1)
#         psis    = psis.view (in_channels, 1, 1, 1, psis.shape[1], 1, 1, 1)

#         thetas  = thetas.view(1,1, 1, 1, 1, thetas.shape[0], 1, 1)
#         y       = y.view(1,1, 1, 1, 1, 1, y.shape[0], 1)
#         x       = x.view(1,1, 1, 1, 1, 1, 1, x.shape[0])

#         sigma_x = sigmas
#         sigma_y = sigmas / gammas

#         sin_t = torch.sin(thetas)
#         cos_t = torch.cos(thetas)
#         y_theta = -x * sin_t + y * cos_t
#         x_theta =  x * cos_t + y * sin_t
        


#         gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
#              * torch.cos(2.0 * math.pi  * x_theta / lambdas + psis)

#         gb = gb.view(-1,kernel_size[0], kernel_size[1])

#         return gb


#     def forward(self, x):
#         batch_size = x.size(0)
#         sy = x.size(2)
#         sx = x.size(3)  
#         gb = self.make_gabor_filters()

#         assert gb.shape[0] == self._n_channels_post_conv
#         assert gb.shape[1] == self.kernel_size
#         assert gb.shape[2] == self.kernel_size
#         gb = gb.view(self._n_channels_post_conv,1,self.kernel_size,self.kernel_size)

#         res = nn.functional.conv2d(input=x, weight=gb,
#             padding=self.kernel_radius, groups=self.in_channels)
       
        
#         if self.rotation_invariant:
#             res = res.view(batch_size, self.in_channels, -1, self.n_thetas,sy, sx)
#             res,_ = res.max(dim=3)

#         res = res.view(batch_size, -1,sy, sx)


#         return res

# def _sigma_prefactor(bandwidth):
#     b = bandwidth
#     # See http://www.cs.rug.nl/~imaging/simplecell.html
#     return 1.0 / 3.14 * torch.sqrt(torch.log(2) / 2.0) * \
#         (2.0 ** b + 1) / (2.0 ** b - 1)


# def torch_gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
#                  n_stds=3, offset=0,lambdas=1):

#     if sigma_x is None:
#         sigma_x = _sigma_prefactor(bandwidth) / frequency
#     if sigma_y is None:
#         sigma_y = _sigma_prefactor(bandwidth) / frequency

#     x0 = torch.ceil(max(torch.abs(n_stds * sigma_x * torch.cos(theta)),
#                      torch.abs(n_stds * sigma_y * torch.sin(theta)), 1))
#     y0 = torch.ceil(max(torch.abs(n_stds * sigma_y * torch.cos(theta)),
#                      torch.abs(n_stds * sigma_x * torch.sin(theta)), 1))
#     y_mes = torch.linspace(-y0,y0+1)
#     x_mes = torch.linspace(-x0,x0 + 1)
#     y, x = torch.meshgrid(y_mes,x_mes)

#     rotx = x * torch.cos(theta) + y * torch.sin(theta)
#     roty = -x * torch.sin(theta) + y * torch.cos(theta)

#     g = torch.zeros(y.shape)
#     g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    
#     g /= 2 * 3.14 * sigma_x * sigma_y
#     g *= torch.exp(2 * np.pi * frequency * rotx + offset)
    
#     # g *= torch.exp(1j * (2 * 3.14 * frequency * rotx + offset))

#     return g
    
# def gabor(sigma, theta, Lambda, psi, gamma):
#     """Gabor feature extraction."""
#     sigma= torch.tensor(sigma)
#     theta = torch.tensor(theta)
#     Lambda = torch.tensor(Lambda)
#     psi = torch.tensor(psi)
#     gamma = torch.tensor(gamma)

#     sigma_x = sigma
#     sigma_y = sigma.float() / gamma

#     # Bounding box
#     nstds = torch.tensor(3)  # Number of standard deviation sigma
#     xmax = torch.abs(nstds * sigma_x * torch.cos(theta)), torch.abs(nstds * sigma_y * torch.sin(theta))
#     xmax = torch.ceil(max(1, xmax))
#     ymax = torch.abs(nstds * sigma_x * torch.sin(theta)), torch.abs(nstds * sigma_y * torch.cos(theta))
#     ymax = torch.ceil(max(1, ymax))
#     xmin = -xmax
#     ymin = -ymax
#     y_mes = torch.linspace(ymin,ymax+1)
#     x_mes = torch.linspace(xmin,xmax + 1)
    
#     y, x = torch.meshgrid(y_mes,x_mes)

#     # Rotation
#     x_theta = x * torch.cos(theta) + y * torch.sin(theta)
#     y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

#     gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * 3.14 / Lambda * x_theta + psi)
#     return gb


