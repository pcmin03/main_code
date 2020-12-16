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
class pretrain_MTL(nn.Module):
    
    def __init__(self,in_channels=1,classes=1,active='sigmoid'):
        super(pretrain_MTL,self).__init__()
        self.model = smp.Unet('resnet34',in_channels=1,classes=1,activation=None,encoder_weights=None)
        
        
        self.decoders = list(self.model.decoder.children())[1]
        
        
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

    def decoder_output(self,decoderlist,encoders):
        class_layer = encoders[-1]
        result = torch.cat([encoders[-2],self.upsample(encoders[5])],1)
        d0 = decoderlist[0](result) 
        d0 = torch.cat([encoders[-3],d0],1)
        d1 = decoderlist[1](d0) 
        d1 = torch.cat([encoders[-4],d1],1)
        d2 = decoderlist[2](d1)
        d2 = torch.cat([encoders[-5],d2],1)
        d3 = decoderlist[3](d2)

        return d3
        

    def forward(self,x):
        encoders = self.encoder_forward(x)

        last = self.decoder_output(self.decoders,encoders)

        backlast = self.sigmoid(self.backlast(last))
        bodylast = self.sigmoid(self.bodylast(last))
        dendlast = self.sigmoid(self.dendlast(last))
        axonlast = self.sigmoid(self.axonlast(last))
        

        # print(backlast.shape,bodylast.shape,dendlast.shape,axonlast.shape)
        
        result = torch.cat((backlast,bodylast,dendlast,axonlast),1)
        
        if self.active == 'sigmoid' or self.active == 'softmax':
            result = result
        else: 
            result = result

        return result,result 

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
