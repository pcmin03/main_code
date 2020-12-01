import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import segmentation_models_pytorch as smp

from torch.autograd import Variable

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
        print(x.shape)
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
                 upsampling_method="conv_transpose",train='layer0'):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if train=='layer0':
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels ,kernel_size=2, stride=2)
        else : 
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels ,kernel_size=2, stride=2,padding=1)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        
        print(up_x.shape,down_x.shape,'11111')
        x = self.upsample(up_x)
        print(x.shape,'2222',down_x.shape,)
        # print(x.shape, 'concat')
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class Multi_UNet(nn.Module):

    def __init__(self, n_class=2):
        super().__init__()
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.body_up3 = double_conv(256 + 512, 256)
        self.body_up2 = double_conv(128 + 256, 128)
        self.body_up1 = double_conv(128 + 64, 64)
        self.body_last = nn.Conv2d(64, n_class, 1)
        
        self.dend_up3 = double_conv(256 + 512, 256)
        self.dend_up2 = double_conv(128 + 256, 128)
        self.dend_up1 = double_conv(128 + 64, 64)
        self.dend_last = nn.Conv2d(64, n_class, 1)
        
        self.axon_up3 = double_conv(256 + 512, 256)
        self.axon_up2 = double_conv(128 + 256, 128)
        self.axon_up1 = double_conv(128 + 64, 64)
        self.axon_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        x = self.first(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        middle = self.dconv_down4(x)

        x = self.upsample(middle)        
        x = torch.cat([x, conv3], dim=1)
        x = self.body_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.body_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.body_up1(x)
        body = self.body_last(x)

        x = self.upsample(middle)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dend_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.dend_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.dend_up1(x)
        dend = self.dend_last(x)

        x = self.upsample(middle)        
        x = torch.cat([x, conv3], dim=1)
        x = self.axon_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.axon_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.axon_up1(x)
        axon = self.axon_last(x)
        
        return body, dend, axon

class multi_Seg_UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(multi_Seg_UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.first = nn.Conv2d(1,3,3,1,padding=1)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.center = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        # self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs = self.first(inputs)
        conv1 = self.conv1(inputs)           # 16*512*512
        maxpool1 = self.maxpool(conv1)       # 16*256*256
        
        conv2 = self.conv2(maxpool1)         # 32*256*256
        maxpool2 = self.maxpool(conv2)       # 32*128*128

        conv3 = self.conv3(maxpool2)         # 64*128*128
        maxpool3 = self.maxpool(conv3)       # 64*64*64

        # conv4 = self.conv4(maxpool3)         # 128*64*64
        # maxpool4 = self.maxpool(conv4)       # 128*32*32

        center = self.center(maxpool3)       # 256*32*32
        print(center.shape)
        up4 = self.up_concat4(center,conv4)  # 128*64*64
        up3 = self.up_concat3(up4,conv3)     # 64*128*128
        up2 = self.up_concat2(up3,conv2)     # 32*256*256
        up1 = self.up_concat1(up2,conv1)     # 16*512*512

        final = self.final(up1)

        return final
    
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
class reconstruction_discrim(nn.Module):
    def __init__(self,in_channels=1,classes=1,multi_output=False):
        super(reconstruction_discrim,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=in_channels,classes=classes,activation='softmax',encoder_weights=None)
        

    def forward(self,x):
        result = self.model.encoder.forward(x)
        final=result[len(result)-1]
        final = final.view(1,-1)
        return final



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
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True),resblock=False):
        super(VGGBlock, self).__init__()
        self.resblock = resblock
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.resblock == True:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act_func(out)
            out = x+out

            out2 = self.conv2(out)
            out2 = self.bn2(out2)
            out2 = self.act_func(out2)
            out2 = out+out2
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act_func(out)
            
            out2 = self.conv2(out)
            out2 = self.bn2(out2)
            out2 = self.act_func(out2)
           
        return out2


class NestedUNet(nn.Module):
    def __init__(self, input_channels=1,out_channels=4,deepsupervision=True,active='sigmoid'):
        super().__init__()
        self.out_channels = out_channels
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

        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()

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
            result = self.sigmoid(output)
            return result,output 

class clas_pretrain_unet(nn.Module):
    def __init__(self,in_channels=1,out_channels=5):
        super(clas_pretrain_unet,self).__init__()
        feature = [64,128,256,512]
        self.model = smp.Unet('resnet34',in_channels=in_channels,classes=out_channels,encoder_weights=None)
        self.decoders = list(self.model.decoder.children())[1]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.classconv = nn.Conv2d(512,out_channels,7,7)
        # print(list(self.model.encoder.children())[-2][2])
        # print(self.model.encoder)
        
        self.classLiner = nn.Linear(feature[-1]*4*4,out_channels)

        self.last = nn.Conv2d(32,out_channels,1)
        # self.classliner = nn.Linear
    def encoder_forward(self, x):
        return self.model.encoder.forward(x)
    
    def forward(self,x,phase='train'):
        encoders = self.encoder_forward(x)
        
        class_layer = encoders[-1]
        # print(encoders[-1].shape,'22',class_layer.view(class_layer.size(0),-1).shape)
        
        # class_feature = F.linear(class_layer.view(class_layer.size(0),-1),(512*(2**len(encoders))*(2**len(encoders)))))
        result = torch.cat([encoders[-2],self.upsample(encoders[5])],1)
        
        d0 = self.decoders[0](result) 
        
        d0 = torch.cat([encoders[-3],d0],1)
        d1 = self.decoders[1](d0) 
        
        d1 = torch.cat([encoders[-4],d1],1)
        d2 = self.decoders[2](d1)

        d2 = torch.cat([encoders[-5],d2],1)
        
        d3 = self.decoders[3](d2)
        
        result = self.last(d3)
        # result = self.softmax(self.finals(d3))
        if phase == 'train':
            class_feature = self.classLiner(class_layer.view(class_layer.size(0),-1))
            return result,F.softmax(class_feature)
        else:
            return result,None
###load model 
class pretrain_unet(nn.Module):
    
    def __init__(self,in_channels=1,classes=4,active='sigmoid'):
        super(pretrain_unet,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=1,classes=classes,activation=None,encoder_weights=None)
        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()
        self.active = active
    def forward(self,x):
        x = self.model(x)
        if self.active == 'sigmoid' or self.active == 'softmax':
            result = self.sigmoid(x)
        else: 
            result = x

        return result,x 
class pretrain_multi_unet(nn.Module):
    
    def __init__(self,in_channels=1,classes=1,active='sigmoid'):
        super(pretrain_multi_unet,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=1,classes=1,activation=None,encoder_weights=None)
        self.body_mo = smp.Unet('resnet34',in_channels=1,classes=1,activation=None,encoder_weights=None)
        self.dend_mo = smp.Unet('resnet34',in_channels=1,classes=1,activation=None,encoder_weights=None)
        self.axon_mo = smp.Unet('resnet34',in_channels=1,classes=1,activation=None,encoder_weights=None)
        
        self.decoders = list(self.model.decoder.children())[1]
        self.body_dec = list(self.body_mo.decoder.children())[1]
        self.dend_dec = list(self.dend_mo.decoder.children())[1]
        self.axon_dec = list(self.axon_mo.decoder.children())[1]
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.backlast = nn.Conv2d(32,1,1)
        self.bodylast = nn.Conv2d(32,1,1)
        self.dendlast = nn.Conv2d(32,1,1)
        self.axonlast = nn.Conv2d(32,1,1)

        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()
        self.active = active

    def encoder_forward(self, x):
        return self.model.encoder.forward(x)

    def decoder_output(self,decoderlist,encoders,lastlayer):
        class_layer = encoders[-1]
        result = torch.cat([encoders[-2],self.upsample(encoders[5])],1)
        d0 = decoderlist[0](result) 
        d0 = torch.cat([encoders[-3],d0],1)
        d1 = decoderlist[1](d0) 
        d1 = torch.cat([encoders[-4],d1],1)
        d2 = decoderlist[2](d1)
        d2 = torch.cat([encoders[-5],d2],1)
        d3 = decoderlist[3](d2)

        return lastlayer(d3)
        

    def forward(self,x):
        encoders = self.encoder_forward(x)

        backlast = self.sigmoid(self.decoder_output(self.decoders,encoders,self.backlast))
        bodylast = self.sigmoid(self.decoder_output(self.body_dec,encoders,self.bodylast))
        dendlast = self.sigmoid(self.decoder_output(self.dend_dec,encoders,self.dendlast))
        axonlast = self.sigmoid(self.decoder_output(self.axon_dec,encoders,self.axonlast))

        # print(backlast.shape,bodylast.shape,dendlast.shape,axonlast.shape)
        
        result = torch.cat((backlast,bodylast,dendlast,axonlast),1)
        
        if self.active == 'sigmoid' or self.active == 'softmax':
            result = result
        else: 
            result = result

        return result,x 

class garbor_pretrain_unet(nn.Module):
    
    def __init__(self,in_channels=4,classes=4,active='sigmoid'):
        super(garbor_pretrain_unet,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=in_channels,classes=classes,activation=None,encoder_weights=None)
        self.lastlayer = nn.Conv2d(4,300,kernel_size=1)
        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()    
        
        
    def forward(self,x,state='train'):
        x = self.model(x)
        if state == 'train':
            # print(x.shape,'xshape')
            mid_x = self.lastlayer(x)
            # print(mid_x.shape,'xshape')
            result = self.sigmoid(x)
            return result,x,mid_x
        else : 
            result = self.sigmoid(x)
            return result,x

# def pretrain_unet(in_channels,classnum):
#     return smp.Unet('resnet34',in_channels=1,classes=classnum,activation='softmax')
    

def pretrain_efficent_unet():
    return smp.Unet('efficientnet-b3',in_channels=1,classes=4,activation='softmax')


class pretrain_deeplab_unet(nn.Module):
    def __init__(self,in_channels=1,classes=4,plus = True,active='sigmoid'):
        super(pretrain_deeplab_unet,self).__init__()
        if plus == True:
            self.model = smp.DeepLabV3Plus('resnet34',in_channels=in_channels,classes=classes,activation=None,encoder_weights=None)
        elif plus == False:
            self.model = smp.DeepLabV3('resnet34',in_channels=in_channels,classes=classes,activation=None,encoder_weights=None)
        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.model(x)
        result = self.sigmoid(x)
        return result,x 

class pretrain_pspnet(nn.Module):
    def __init__(self,in_channels=1,classes=4,active='sigmoid'):
        super(pretrain_pspnet,self).__init__()
        self.model = smp.PSPNet('resnet34',in_channels=in_channels,classes=classes,activation=None)
        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.model(x)
        result = self.sigmoid(x)
        return result,x 

class refinenet(nn.Module):
    def __init__(self,in_channels=1,classes=4,multi_output=True):
        super(refinenet,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=in_channels,classes=classes,activation='softmax',encoder_weights=None)
        feature = [64,128,256,512]
        self.multi_output = multi_output
        self.first    = single_conv(1,3) #1024
        self.encoder0 = single_conv(3,feature[0]) #512
        self.encoder1 = single_conv(feature[0],feature[0]) #256
        self.encoder2 = single_conv(feature[0],feature[1]) #128
        self.encoder3 = single_conv(feature[1],feature[2]) #64
        self.encoder4 = single_conv(feature[2],feature[3]) #32

        self.model_0 = VGGBlock(feature[0],feature[0],feature[0],resblock=True)
        self.model_1 = VGGBlock(feature[0],feature[0],feature[0],resblock=True)
        self.model_2 = VGGBlock(feature[1],feature[1],feature[1],resblock=True)
        self.model_3 = VGGBlock(feature[2],feature[2],feature[2],resblock=True)
        self.model_4 = VGGBlock(feature[3],feature[3],feature[3],resblock=True)
        
        # self.model_0 = smp.Unet('resnet34',in_channels=64,classes=64,activation='softmax',encoder_weights=None)
        # self.model_1 = smp.Unet('resnet34',encoder_depth=4,in_channels=64,classes=64,activation='softmax',encoder_weights=None)
        # self.model_2 = smp.Unet('resnet34', encoder_depth=4,in_channels=128,classes=128,activation='softmax',encoder_weights=None)
        # self.model_3 = smp.Unet('resnet34', encoder_depth=4,in_channels=256,classes=256,activation='softmax',encoder_weights=None)
        # self.model_4 = smp.Unet('resnet34', encoder_depth=4,in_channels=512,classes=512,activation='softmax',encoder_weights=None)
        

        self.deconv5,self.deconv4,self.deconv3,self.deconv2,self.deconv1 = list(self.model.decoder.children())[1]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.finals = nn.Conv2d(32, classes, kernel_size=3,padding=1)
        
        self.finals_d2 = nn.Conv2d(feature[1], classes, kernel_size=3,padding=1)
        self.finals_d1 = nn.Conv2d(feature[1]+feature[0], classes, kernel_size=3,padding=1)
        self.finals_d0 = nn.Conv2d(feature[1]+feature[2], classes, kernel_size=3,padding=1)
        
        
        self.softmax = nn.Softmax(dim=1)
        # print(list(self.model.decoder.children())[1])
    def pre_forward(self,x):
        return self.model.encoder.forward(x)
    
    def pre_forwardv2(self,x):
        e0 = self.first(x)
        # print(e0.shape)
        # feautre_size = x.shape[0,:]
        e1 = F.avg_pool2d(self.encoder0(e0),2)
        e2 = F.avg_pool2d(self.encoder1(e1),2)
        e3 = F.avg_pool2d(self.encoder2(e2),2)
        e4 = F.avg_pool2d(self.encoder3(e3),2)
        e5 = F.avg_pool2d(self.encoder4(e4),2)
        
        return [e0,e1,e2,e3,e4,e5]
    
    def forward(self,x):
        encoder_list =self.pre_forwardv2(x)
        # print()
        # orch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32])
# torch.Size([1, 3, 128, 128]) torch.Size([1, 64, 64, 64]) torch.Size([1, 64, 32, 32]) torch.Size([1, 128, 16, 16]) torch.Size([1, 256, 8, 8]) torch.Size([1, 512, 4, 4])

        # print(encoder_list[1].shape,encoder_list[2].shape,encoder_list[3].shape,encoder_list[4].shape,encoder_list[5].shape)
        # print(len(encoder_list))
        encoder_list[2] = self.model_1(encoder_list[2])
        encoder_list[3] = self.model_2(encoder_list[3])
        encoder_list[4] = self.model_3(encoder_list[4])
        encoder_list[5] = self.model_4(encoder_list[5])

        
        # print(encoder_list[1].shape,encoder_list[2].shape,encoder_list[3].shape,encoder_list[4].shape,encoder_list[5].shape)
        # print(list(self.model.decoder.children())[1])
        # print(encoder_list[5].shape)
        result = torch.cat([encoder_list[4],self.upsample(encoder_list[5])],1)
        d0 = self.deconv5(result) 
        # print(result.shape)
        d0 = torch.cat([encoder_list[3],d0],1)
        d1 = self.deconv4(d0) 
        
        d1 = torch.cat([encoder_list[2],d1],1)
        d2 = self.deconv3(d1)

        d2 = torch.cat([encoder_list[1],d2],1)
        
        d3 = self.deconv2(d2)
        result = self.softmax(self.finals(d3))

        if self.multi_output == True:
            d0 = self.softmax(self.finals_d0(d0))
            d1 = self.softmax(self.finals_d1(d1))
            d2 = self.softmax(self.finals_d2(d2))
            return [result,d2,d1,d0]
        else:
            return result

class SideOutput(nn.Module):

    def __init__(self, num_output, out_class=20,kernel_sz=None, stride=None):
        super(SideOutput, self).__init__()
        self.conv = nn.Conv2d(num_output, out_class, 1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.Conv2d(out_class,out_class,3,padding=1)
        else:
            self.upsample = False

    def forward(self, res, size):
        side_output = self.conv(res)
        side_output = self.upsampled(F.upsample(side_output,size=size))

        return side_output

class Res5Output(nn.Module):

    def __init__(self, num_output=2048, out_class=20,kernel_sz=8, stride=8):
        super(Res5Output, self).__init__()
        self.conv = nn.Conv2d(num_output, out_class, 1, stride=1, padding=0)
        # self.upsampled = nn.ConvTranspose2d(out_class, out_class, kernel_size=kernel_sz, stride=stride, padding=0)
        # self.upsampled = nn.Upsample(size=(1024,1024),scale_factor=(2,2),mode='bilinear')
        self.upsampled = nn.Conv2d(out_class,out_class,3,padding=1)
        
        
    def forward(self, res,size):
        res = self.conv(res)
        res = self.upsampled(F.upsample(res,size=size))
        return res

class CASENet(nn.Module):
    def __init__(self,in_channels=1,classes=4,multi_output=True):
        super(CASENet,self).__init__()
        self.model = smp.Unet('resnet50',in_channels=in_channels,classes=classes,activation='softmax',encoder_weights=None)
        feature = [64,128,256,512]
        feature = [64,128,256,512,1024,2048]
        self.SideOutput_e5 = SideOutput(feature[5], kernel_sz=4, stride=2)
        self.SideOutput_e4 = SideOutput(feature[3], kernel_sz=4, stride=2)
        self.SideOutput_e3 = SideOutput(feature[2], kernel_sz=4, stride=2)
        self.Res5Output_e0 = Res5Output(feature[0])
        
    def pre_forward(self,x):
        return self.model.encoder.forward(x)
    

    def _sliced_concat(self, res1, res2, res3, res5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(res1.size(0), out_dim, res1.size(2), res1.size(3))).cuda()
        for i in range(0, out_dim, 4):
            class_num = 0
            out_tensor[:, i, :, :] = res1[:, class_num, :, :]
            out_tensor[:, i + 1, :, :] = res2[:, class_num, :, :]
            out_tensor[:, i + 2, :, :] = res3[:, class_num, :, :]
            out_tensor[:, i + 3, :, :] = res5[:, class_num, :, :]
            class_num += 1

        return out_tensor

    def _fused_class(self, sliced_cat, groups):
        in_channels = sliced_cat.size(1)
        out_channels = sliced_cat.size(1)//groups
        conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups).cuda()
        out = conv(sliced_cat).cuda()
        return out
        
    def forward(self,x):
        original_size = x.size()[2:]
        encoders = self.pre_forward(x)
        # print(len(encoders))
        # print(encoders[5].shape,encoders[4].shape,encoders[3].shape,encoders[2].shape,encoders[1].shape)
        encoders[5] = self.SideOutput_e5(encoders[5],original_size) # 512
        encoders[4] = self.SideOutput_e4(encoders[3],original_size) # 256
        encoders[3] = self.SideOutput_e3(encoders[2],original_size) # 128
        encoders[2] = self.Res5Output_e0(encoders[1],original_size) # 64
        
        sliced_cat = self._sliced_concat(encoders[2], encoders[3], encoders[4], encoders[5], 4)
        acts = self._fused_class(sliced_cat, 4)
        return acts

class classification_model(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.first = nn.Conv2d(4,3,3,1,padding=1)
        self.classfier = models.resnet18(pretrained=False)

        self.lelu = nn.LeakyReLU(0.2, inplace=True)
        self.last = nn.Linear(1000, 512)
        self.lelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.last2 = nn.Linear(512, 1)

    def forward(self,x):
        x = self.first(x)
        x = self.classfier(x)
        # print(x.shape)
        x = self.last(self.lelu(x))
        x = self.last2(self.lelu2(x))
        # print(x.shape)
        # x = F.sigmoid(x)
        return x


        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True)
            ,nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    # 
    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=4):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class efficient_nestunet(nn.Module):
    def __init__(self,in_channels,classes,multi_output=False):
        super(efficient_nestunet,self).__init__()
        #unet forward
        feature_ = [40,32,48,136,384]

        self.model =smp.Unet('efficientnet-b3',in_channels=1,classes=4)
        
        delayers = list(self.model.decoder.children())[1]
        self.deconv5,self.deconv4,self.deconv3,self.deconv2,self.deconv1 = delayers
        # print(self.deconv1)
        self.deconv1 = VGGBlock(feature_[1],16,16)
        # self.finals = self.model.segmentation_head
        
        self.up_x2_1 = single_conv(feature_[3]+feature_[2]  ,feature_[2])
        self.up_x1_2 = single_conv(feature_[2]+feature_[1]*2,feature_[1])
        self.up_x0_3 = single_conv(feature_[1]+feature_[0]*3,feature_[0])
        self.topconcat = single_conv(feature_[0]*4,feature_[0])

        self.up_x1_1 = single_conv(feature_[2]+feature_[1]  ,feature_[1])
        self.up_x0_2 = single_conv(feature_[1]+feature_[0]*2,feature_[0])
        self.midconcat = single_conv(feature_[1]*3,feature_[1])

        self.up_x0_1 = single_conv(feature_[1]+feature_[0]  ,feature_[0])
        self.lastconcat = single_conv(feature_[2]*2,feature_[2])
        
        if multi_output == True:
            self.final1 = nn.Conv2d(feature_[0], classes, kernel_size=1)
            self.final2 = nn.Conv2d(feature_[0], classes, kernel_size=1)
            self.final3 = nn.Conv2d(feature_[0], classes, kernel_size=1)
        self.multi_output = multi_output
    
        self.finals = nn.Conv2d(16, classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def encoderforward(self,x):
        return self.model.encoder.forward(x)
    def forward(self,x):
        _,x0_0,x1_0,x2_0,x3_0,x4_0 = self.encoderforward(x)
        
        #first skip connection
        x0_1 = self.up_x0_1(torch.cat([x0_0,self.upsample(x1_0)],1))

        #second skip connection
        x1_1 = self.up_x1_1(torch.cat([x1_0,self.upsample(x2_0)],1))
        x0_2 = self.up_x0_2(torch.cat([x0_0,x0_1,self.upsample(x1_1)],1))

        #third skip connection
        x2_1 = self.up_x2_1(torch.cat([x2_0,self.upsample(x3_0)],1))
        x1_2 = self.up_x1_2(torch.cat([x1_0,x1_1,self.upsample(x2_1)],1))
        x0_3 = self.up_x0_3(torch.cat([x0_0,x0_1,x0_2,self.upsample(x1_2)],1))

        #forth depth
        x3_1 = torch.cat([x3_0,self.upsample(x4_0)],1)
        x3_1 = self.deconv5(x3_1)

        #third depth
        x2_0 = self.lastconcat(torch.cat([x2_0,x2_1],1))
        x2_2 = torch.cat([x2_0,x3_1],1)
        x2_2 = self.deconv4(x2_2)

        #second depth
        x1_0 = self.midconcat(torch.cat([x1_0,x1_1,x1_2],1))
        x1_3 = torch.cat([x1_0,x2_2],1)
        x1_3 = self.deconv3(x1_3)

        #first depth
        x0_0 = self.topconcat(torch.cat([x0_0,x0_1,x0_2,x0_3],1))
        x0_4 = torch.cat([x0_0,x1_3],1)
        x0_4 = self.deconv2(x0_4)
        
        last = self.deconv1(x0_4)
        # print(last.shape,'111')
        last = self.finals(last)
        
        
        if self.multi_output == True:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            result= self.finals(last)
            return [output1,output2,output3,result]
        return self.softmax(last)
