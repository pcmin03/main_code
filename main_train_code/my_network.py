import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import segmentation_models_pytorch as smp

# import resnet
# from resnet import *
# from resnet import *
#=====================================================================#
#===========================resunet network===========================#
#=====================================================================#
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWithResNet50(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class Segmentataion_resnet101unet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=4):
        super().__init__()
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        resnet = models.resnet.resnet101(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        x = self.first(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Segmentataion_resnet101unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x


        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Segmentataion_resnet101unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

#=====================================================================#
#==========================multiple network===========================#
#=====================================================================#

class multiUpBlockForUNetWithResNet101(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class multi_Segmentataion_resnet101unet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        # self.first = nn.Conv2d(1,3,3,1,padding=1)
        resnet = models.resnet.resnet101(pretrained=True)
        down_blocks = []
        global_up_blocks = []
        dend_up_blocks = []
        axon_up_blocks = []

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        global_up_blocks.append(multiUpBlockForUNetWithResNet101(2048, 1024))
        global_up_blocks.append(multiUpBlockForUNetWithResNet101(1024, 512))
        global_up_blocks.append(multiUpBlockForUNetWithResNet101(512, 256))
        global_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        global_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.global_up_blocks = nn.ModuleList(global_up_blocks)

        dend_up_blocks.append(multiUpBlockForUNetWithResNet101(2048, 1024))
        dend_up_blocks.append(multiUpBlockForUNetWithResNet101(1024, 512))
        dend_up_blocks.append(multiUpBlockForUNetWithResNet101(512, 256))
        dend_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        dend_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.dend_up_blocks = nn.ModuleList(dend_up_blocks)

        axon_up_blocks.append(multiUpBlockForUNetWithResNet101(2048, 1024))
        axon_up_blocks.append(multiUpBlockForUNetWithResNet101(1024, 512))
        axon_up_blocks.append(multiUpBlockForUNetWithResNet101(512, 256))
        axon_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        axon_up_blocks.append(multiUpBlockForUNetWithResNet101(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.axon_up_blocks = nn.ModuleList(axon_up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False,select_training='full'):
        # x = self.first(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Segmentataion_resnet101unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x


        x = self.bridge(x)
        
        if select_training == 'ful':
            for i, block in enumerate(self.global_up_blocks, 1):
                key = f"layer_{Segmentataion_resnet101unet.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])
            output_feature_map = x
            x = self.out(x)
            return x

        elif select_training == 'dend':
            for i, block in enumerate(self.dend_up_blocks, 1):
                key = f"layer_{Segmentataion_resnet101unet.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])
            output_feature_map = x
            x = self.out(x)
            return x
        
        elif select_training == 'axon':
            for i, block in enumerate(self.axon_up_blocks, 1):
                key = f"layer_{Segmentataion_resnet101unet.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])
            output_feature_map = x
            x = self.out(x)
            return x

#=====================================================================#
#=======================3d segmentation network=======================#
#=====================================================================#

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        # self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        # print(x.shape)
        out = self.bn1(self.conv1(x))
        # print(out.shape)
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=1)
        self.relu1 = ELUCons(elu, 4)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # print(out.shape)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        # self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        # self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        # print(out128.shape)
        # out256 = self.down_tr256(out128)
        # out = self.up_tr256(out128, out128)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
#=====================================================================#
#==========================3d resnet network==========================#
#=====================================================================#

class ConvBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Bridge_3D(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ConvBlock3D(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class UpBlockForUNetWith3DResNet101(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="deconv"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels
        if upsampling_method =='skernel':
            self.upsample = nn.ConvTranspose3d(up_conv_in_channels, up_conv_out_channels, kernel_size=(1,2,2), stride=(1,2,2))
        elif upsampling_method == 'deconv': 
            self.upsample = nn.ConvTranspose3d(up_conv_in_channels, up_conv_out_channels, kernel_size=(2,2,2), stride=(2,2,2))
        else:
            self.upsample = nn.ConvTranspose3d(up_conv_in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1)) 
        self.conv_block_1 = ConvBlock3D(in_channels, out_channels)    
        self.conv_block_2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        # print(up_x.shape,down_x.shape)
        x = self.upsample(up_x)
        # print(x.shape,down_x.shape)
        x = torch.cat([x, down_x], 1)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class unet3d_resnet(nn.Module):
    DEPTH = 5

    def __init__(self, n_classes=4):
        super().__init__()
        # self.first = nn.Conv2d(1,3,3,1,padding=1)
        resnet = resnet101(num_classes=1,
                shortcut_type='B',
                sample_size=256,
                sample_duration=8)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children())[:7]:
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        # self.bridge = Bridge_3D(2048, 2048)
        # up_blocks.append(UpBlockForUNetWith3DResNet101(2048, 1024,upsampling_method='skernel'))
        up_blocks.append(UpBlockForUNetWith3DResNet101(1024, 512,upsampling_method='skernel'))
        up_blocks.append(UpBlockForUNetWith3DResNet101(512, 256,upsampling_method='deconv'))
        up_blocks.append(UpBlockForUNetWith3DResNet101(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128,upsampling_method='deconv'))
        up_blocks.append(UpBlockForUNetWith3DResNet101(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64,upsampling_method='last'))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv3d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        # x = self.first(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        # print(x.shape)
        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            # print(x.shape)
            if i == (unet3d_resnet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x


        # x = self.bridge(x)/
        # print(x.shape)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{unet3d_resnet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
            # print(x.shape)
        output_feature_map = x
        x = self.out(x)
        
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
#=====================================================================#
#===========================detecion network==========================#
#=====================================================================#
def get_model_instance_segmentation(num_classes=2):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


#=====================================================================#
#==============================U-network==============================#
#=====================================================================#
class block_encoder(nn.Module):
    def __init__(self, in_ch,out_ch,use_padding = True):
        super().__init__()
        if use_padding == True:
            self.encoder = torch.nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
        else : 
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
                            
    def forward(self,x):
        return self.encoder(x)

class block_decoder(nn.Module):
    def __init__(self, in_ch,out_ch,use_deconv = True):
        super().__init__()
        if use_deconv == True:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_ch,out_ch,kernel_size=3,padding=1,output_padding=1,stride = 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
                
            self.decoder_block = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
        else : 
            self.upsample = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
            self.decoder_block = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())

    def forward(self,up_x, down_x):
        x = self.upsample(down_x)
        x = torch.cat([x,up_x],dim=1)
        
        return self.decoder_block(x)

class middle_block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.encoder = torch.nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding = 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())

    def forward(self,x):
        return self.encoder(x)

class Unet(nn.Module):
    def __init__(self, in_ch,out_ch=4,depth=5):
        super().__init__()
        down_blocks = []
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        feature_list = [64,128,256,512,1024]
        for i in range(depth):
            if i == 0 : 
                down_blocks.append(block_encoder(in_ch ,feature_list[i]))
            else:
                down_blocks.append(block_encoder(feature_list[i-1],feature_list[i]))
        
        self.down_blocks = nn.ModuleList(down_blocks)
        
        self.block = middle_block(feature_list[i],feature_list[i]) 
        
        up_blocks = []
        for j in range(depth):
            
            if i == 0: 
                up_blocks.append(block_decoder(feature_list[i],feature_list[i-1]))
            else: 
            
                up_blocks.append(block_decoder(feature_list[i],feature_list[i-1]))
                i -= 1
        self.up_blocks = nn.ModuleList(up_blocks)

        self.last_block = nn.Conv2d(feature_list[i],out_ch,kernel_size=1,padding=0)
        
    def forward(self,x):
        
        #encoder
        en_list = dict()  
        for i, ENblock in enumerate(self.down_blocks):
            x = ENblock(x)           
            en_list[f"layer_{i}"] = x
            
            if i < self.depth-1:
                x = self.pool(x)
            
        x = self.block(x)
        
        #decoder
        for j, DEblock in enumerate(self.up_blocks):
            i -=1 
            if i >= 0:
                x = DEblock(en_list[f"layer_{i}"],x)
            # x = DEblock(x)
        
        return self.last_block(x)

#=====================================================================#
#==============================U-network==============================#
#=====================================================================#

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class NestedUNet(nn.Module):
    def __init__(self, input_channels=1,out_channels=4,deepsupervision=True):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.deepsupervision = deepsupervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision == True:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision == True:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


###load model 
def pretrain_unet():
    return smp.Unet('resnet34',in_channels=1,classes=4,activation='softmax')
def pretrain_efficent_unet():
    return smp.Unet('efficientnet-b5',in_channels=1,classes=4,activation='softmax')

class classification_model(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        self.classfier = models.resnet34(pretrained=True)
    def forward(self,x):
        x = self.first(x)
        return self.classfier(x)
# def compute_gradient_penalty(netD, real_data, fake_data):
    
#     # print "real_data: ", real_data.size(), fake_data.size()
#     alpha = Variable(torch.rand(1),requires_grad=True)

#     alpha = Tensor(np.random.random((real_data.size(0),1, 1, 1))).to(cuda0)

#     # alpha = Variable(torch.rand(BATCH_SIZE,1,1,1),requires_grad=True)
#     alpha = alpha.expand(real_data.size()).to(cuda0)
#     # print(alpha.shape)
#     interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

#     if cuda0:
#         interpolates = interpolates.to(cuda0)
#     interpolates = Variable(interpolates, requires_grad=True)

#     disc_interpolates = netD(interpolates)

#     gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).to(cuda0),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#     # gradients = gradients.view(gradients.size(0), -1)

#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
