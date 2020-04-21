import cv2
import skimage
import torch
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import scipy
import numbers

from numpy.lib.stride_tricks import as_strided

from natsort import natsorted
from nd2reader import ND2Reader
from skimage.util.shape import view_as_blocks
from skimage.filters import threshold_otsu, threshold_yen

import torch.nn.functional as F
from torch.utils.data import  Dataset

from sklearn.model_selection import KFold

#=====================================================================#
#============================pre processing===========================#
#=====================================================================#
def decode_segmap(image, nc=4,name='full'):

    if name =='body':
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=cellbody, 2=dendrite, 3=axon
                    (254, 254, 0)])
    elif name =='dend':
        label_colors = np.array([(0,0,0),(254, 24, 254)])
    elif name =='axon':
        label_colors = np.array([(0,0,0), (0, 146, 146)])
    elif name == 'full':
        label_colors = np.array([(0,0,0),(254,254,0),(254,24,254),(0,146,146)])
    elif name == 'inverse':
        label_colors = np.array([(0,0,0),(254,254,0),(254,254,0),(254,254,0)])
    # print(label_colors.shape)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=1)
    return rgb

def channel_wise_segmentation(image, nc=4):

    label_colors = np.array([(0,0,0),(254,254,0),(254,24,254),(0,146,146)])
    project_image = 0
    
    print(image.max(),'123')
    for i in range(0,nc):        
        r = np.zeros_like(image[:,i]).astype(np.uint8)
        g = np.zeros_like(image[:,i]).astype(np.uint8)
        b = np.zeros_like(image[:,i]).astype(np.uint8)
        # for l in range(0, nc):
        idx = image == 1
        print(r.max(),'1111')
        r[r==1] = label_colors[i][0]
        g[g==1] = label_colors[i][1]
        b[b==1] = label_colors[i][2]
        rgb = np.stack([r, g, b], axis=1)
        project_image += rgb
        print(r.max())
    # print(rgb.shape,'1111')
    return np.array(project_image)



def multi_decode(multi_image,state):
    stack_img = []
    
    for i in range(len(multi_image[0,:])):
        s_img= decode_segmap(multi_image[:,i],name=state)
        
        stack_img.append(s_img)
    return np.transpose(np.array(stack_img),(1,2,0,3,4))

def ch_channel(img):
    return  torch.argmax(img,dim=1).cpu().detach().numpy()

##############preprocessing#################
def preprocessing(nd2Dir='./#12_3.nd2'):

    normalizedImg = np.zeros((1024, 1024))
    win_size = 256

    with ND2Reader(nd2Dir) as images:
        images.bundle_axes = 'czyx'
        full_images = np.array(images)

    stack_images = full_images[0,0].astype('uint16')
    stack_mitos = full_images[0,1].astype('uint16')
    
    _,xsize,ysize=stack_images.shape
    if xsize > 1024 and ysize > 1024 :
        dim_size = xsize /2
        x_mi = int(dim_size-512)
        x_ma = int(dim_size+512)
        stack_images = stack_images[:,x_mi:x_ma,x_mi:x_ma]
        stack_mitos= stack_mitos[:,x_mi:x_ma,x_mi:x_ma]

    #normalize data#
    pro_img = cv2.normalize(stack_images,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    pro_mito = cv2.normalize(stack_mitos, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    
    #threshold background#
    img_threshold = skimage.filters.threshold_yen(pro_img)
    mito_threshold = skimage.filters.threshold_otsu(pro_mito)
    
    pro_img = (pro_img > img_threshold) * pro_img
    pro_mito = (pro_mito > mito_threshold) * pro_mito


    pro_img = pro_img.astype('uint8')
    pro_mito = pro_mito.astype('uint8')

    depth,_,_=pro_img.shape
    patch_pro_img = view_as_blocks(pro_img,block_shape=(depth,win_size,win_size))
    _,_,num,depth,size,_=patch_pro_img.shape
    patch_pro_img = np.reshape(patch_pro_img,(num*num,depth,size,size))
    
    pro_img = np.max(stack_images,axis=0)
    pro_mito = np.max(pro_mito,axis=0)

    pro_mito = skimage.color.gray2rgb(pro_mito)
    return patch_pro_img, pro_mito

def make_full_image(patch_img):
    new_image = []
    himag = []
    for j in range(len(patch_img)//4):
        full_image=cv2.hconcat([patch_img[j*4],patch_img[j*4+1],patch_img[j*4+2],patch_img[j*4+3]])
        himag.append(full_image)
        if j==0:
            continue
        elif j%3 == 0:
            new=np.array(himag)
            full_image2=cv2.vconcat([new[0],new[1],new[2],new[3]])
            new_image.append(full_image2)
    return np.array(new_image)[0]

def view_as_windows(arr_in, window_shape, step=10,threshold=0.05):
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    num,_,win_shape,_=arr_out.shape
    arr_out = np.reshape(arr_out,[num * num,win_shape,win_shape])
    
    new_image = []

    for i,vlaue in enumerate(arr_out):
        count_num = np.sum(np.array(vlaue) > 0)
        if count_num >= (4096*threshold):
            new_image.append(vlaue)        
    new_image = np.array(new_image)
    
    return new_image


def divide_getlabel(full_image,image,select_channel):
    label2 = skimage.color.rgb2gray(full_image)
        
    if image.ndim == 3:
        image = skimage.color.rgb2gray(image)
    #give label
    mask0= label2 < 0.3
    mask1= label2 > 0.9
    mask2= (label2 >0.3) & (label2 < 0.4)
    mask3= (label2 >0.4) & (label2 < 0.8) 

    label2[mask0] = 0 #background
    label2[mask1] = 0 #soma
    label2[mask2] = 2 #dendrite
    label2[mask3] = 3 #axon
    if select_channel == 'dendrite':
        label2[mask3] = 0
        return scipy.ndimage.morphology.binary_fill_holes(label2/2 * image).astype('uint8')
    elif select_channel == 'axon':
        label2[mask2] = 0
        return scipy.ndimage.morphology.binary_fill_holes(label2/3 * image).astype('uint8')
    elif select_channel == 'full':
        label2[mask2] = 1
        label2[mask3] = 1
        return scipy.ndimage.morphology.binary_fill_holes(label2 * image).astype('uint8')

def locateComponents(img,minSiz=4,maxSiz=50):
    """Extracts all components from an image"""

    out = img.copy()     
    contours, hierarchy = cv2.findContours(np.uint8(out.copy()),\
                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    ret = []
    row, col = out.shape
    for cnt in contours:
        # get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        # check area 
        if w < minSiz or h < minSiz:
            continue
        if w < maxSiz  and h < maxSiz:
            ret.append(np.int32([x, y, x+w, y+h]))
            out = skimage.color.gray2rgb(out).astype('uint8')
    #         out = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)
            out = cv2.rectangle(out, (x,y), (x+w,y+h), (255,255,0), 1)

    return ret, out

def detecion_box(img,predictions):
    boxes = list(predictions)[0]['boxes'].cpu().detach().numpy()
    for box in boxes:
        x,y,m,n = box
        img = skimage.color.gray2rgb(img).astype('uint8')
        img = cv2.rectangle(img, (x,y), (m,n), (255,255,0), 1)
    return img, boxes

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

##############loss#################
class Custom_WeightedCrossEntropyLossV2(torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """
    def forward(self, net_output, gt):
        # print(num_class.)
        # class_weights = self._class_weights(inp)
        new_output = torch.argmax(net_output,dim=1)
        MSEloss = F.mse_loss(new_output.float(),gt.float())
        weight_MSEloss = 1/(1+MSEloss)

        gt = gt.long()
        num_classes = net_output.size()[1]
        i0 = 1
        i1 = 2
        pre = net_output
        while i1 < len(net_output.shape): # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose(i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous()
        net_output = net_output.view(-1, num_classes) #shape=(vox_num, class_num)

        gt = gt.view(-1,)
        # print(net_output.shape,gt.shape)
        BCEloss = F.cross_entropy(net_output ,gt)
        return  BCEloss * weight_MSEloss

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)
        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

##########load dataset###########
# class MYdataset_2d(Dataset):
#     def __init__(self,imageDir,labelDir,transform,L_transform,state):
#         img = glob.glob(imageDir +'*')
#         lab = glob.glob(labelDir +'*')

#         self.images = natsorted(img)
#         self.labels = natsorted(lab)

#         self.transform = transform
#         self.L_transform = L_transform
#         self.state = state
#         print(len(img),len(lab))
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self,index):
        
#         image = skimage.io.imread(self.images[index])
#         label = skimage.io.imread(self.labels[index])

#         image = np.array(image).astype('uint8')
#         label = np.array(label)

#         label = np.max(label,axis=0).astype('uint8')
#         label2 = skimage.color.rgb2gray(label)
        
#         #give label
#         mask0= label2 < 0.3
#         mask1= label2 > 0.9
#         mask2= (label2 >0.3) & (label2 < 0.4)
#         mask3= (label2 >0.4) & (label2 < 0.8) 

#         label2[mask0] = 0
#         label2[mask1] = 1
#         label2[mask2] = 2
#         label2[mask3] = 3
#         image= np.max(image, axis=0)
#         image = skimage.color.gray2rgb(image)
#         if self.state == 'train':
#             toPIL = transforms.ToPILImage()
#             image = toPIL(image.astype('uint8'))
#             label2 = toPIL(label2.astype('uint8'))
#             # image = image.convert('rgb')
#             image,label2 = t_transform([image,label2])
#             if random.random() > 0.5:        
#                 angle = random.randint(0, 3)
#                 roate = tr.RandomRotate( angle*90)
#                 image,label2 = roate([image,label2])
                
#         # image = image.convert('rgb')

#         clip = self.L_transform(image)
#         label2 = np.array(label2)

#         return clip,label2

class custom_mydataset_3d(Dataset):
    def __init__(self,imageDir,labelDir,transform,L_transform,state):
        img = glob.glob(imageDir +'*')
        lab = glob.glob(labelDir +'*')

        self.images = natsorted(img)
        self.labels = natsorted(lab)

        self.transform = transform
        self.L_transform = L_transform
        self.state = state
        print(len(img),len(lab))
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        
        image = skimage.io.imread(self.images[index])
        label = skimage.io.imread(self.labels[index])

        image = np.array(image).astype('uint8')
        label = np.array(label)

        label = np.max(label,axis=0).astype('uint8')
        label2 = skimage.color.rgb2gray(label)
        
        #give label
        mask0= label2 < 0.3
        mask1= label2 > 0.9
        mask2= (label2 >0.3) & (label2 < 0.4)
        mask3= (label2 >0.4) & (label2 < 0.8) 

        label2[mask0] = 0
        label2[mask1] = 1
        label2[mask2] = 2
        label2[mask3] = 3
        
        image= np.max(image, axis=0)
        image = skimage.color.gray2rgb(image)
        if self.state == 'train':
            toPIL = transforms.ToPILImage()
            image = toPIL(image.astype('uint8'))
            label2 = toPIL(label2.astype('uint8'))
            # image = image.convert('rgb')
            image,label2 = t_transform([image,label2])
            if random.random() > 0.5:        
                angle = random.randint(0, 3)
                roate = tr.RandomRotate( angle*90)
                image,label2 = roate([image,label2])
                
        # image = image.convert('rgb')

        clip = self.L_transform(image)
        label2 = np.array(label2)

        return clip,label2
        
