import cv2,skimage
import torch
import numpy as np
# import pydensecrf.densecrf as dcrf
# import pydensecrf.utils as utils
import scipy,os
import numbers, glob
from glob import glob 
from numpy.lib.stride_tricks import as_strided

from natsort import natsorted
from nd2reader import ND2Reader
from skimage.util.shape import view_as_blocks
from skimage.filters import threshold_otsu, threshold_yen
from skimage.morphology import erosion

import torch.nn.functional as F
from torch.utils.data import  Dataset
from torch import nn
from sklearn.model_selection import KFold
from skimage.morphology import square

# natsort
from torch.nn.modules.utils import _pair, _quadruple


#=====================================================================#
#============================pre processing===========================#
#=====================================================================#
def make_path(args):
    #activation name, 
    modelsave_dir = '../'+str(args.modelname)+'_'+str(args.datatype)+'/'
    if not os.path.exists(modelsave_dir):
        print(f' Make_save_Dir : {modelsave_dir}')
        os.makedirs(modelsave_dir)
    valdation_dir = str(args.knum)+str(args.activation)+'_'+str(args.patchsize)+'_'
    if args.oversample == True:
        valdation_dir += 'oversample_'
        
    return modelsave_dir, valdation_dir

def EROSION(image,sigma=6):
    # print(image.shape)
    image[0] =  erosion(image[0],square(sigma))
    print(image.shape)
    return image


def decode_segmap(images, nc=4,name='full_4'):

    if name =='body':
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=cellbody, 2=dendrite, 3=axon
                    (254, 254, 0)])
    elif name =='dend':
        label_colors = np.array([(0,0,0),(254, 24, 254)])
    elif name =='axon':
        label_colors = np.array([(0,0,0), (0, 146, 146)])
    elif name == 'full_4':
        label_colors = np.array([(0,0,0),(254,254,0),(254,24,254),(0,146,146)])
    elif name == 'full':
        label_colors = np.array([(255,255,255),(0,0,0),(254,254,0),(254,24,254),(0,146,146)])
    elif name == 'inverse':
        label_colors = np.array([(0,0,0),(254,254,0),(254,254,0),(254,254,0)])
    elif name == 'convert':
        label_colors = np.array([(255,255,255),(254,254,0),(254,24,254),(0,146,146)])
    # print(label_colors.shape)
    total_rgb = []
    # for image in images[:,:]:
    image = images
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        # idx = (image > 0.5)
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=1)
        
        # total_rgb.append(rgb)
    # print(np.array(total_rgb).shape,'imagesimagesimages')
    return np.array(rgb)
# [0.0721 :background    0.35081569 0.35923216:dendrtie 0.45088235 0.92426118:body 0.9663341:background
#  0.98877804 1.        ] 2323232

def multi_decode(multi_image,state):
    stack_img = []
    
    for i in range(len(multi_image)):
        
        s_img= decode_segmap(multi_image[i:i+1],name=state)[0]
        
        stack_img.append(s_img)
    # print(np.array(stack_img).shape)
    stack_img =  np.expand_dims(np.array(stack_img),axis=1)
    print(stack_img.shape)
    return np.transpose(stack_img,(1,0,4,2,3))

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

def make_full_image(images,s_point = 4): 
    images = images[:16]
    himag =[]
    for j in range(s_point):
        full_image=np.concatenate([images[j*4],images[j*4+1],images[j*4+2],images[j*4+3]],axis=-1)
        himag.append(full_image)

    himag = np.array(himag)
    full_image=np.concatenate([himag[0],himag[1],himag[2],himag[3]],axis=-2)
    full_image = full_image[np.newaxis]
    return full_image


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
            out = cv2.rectangle(out, (x,y), (x+w,y+h), (255,255,0), 1)

    return ret, out

def detecion_box(img,predictions):
    boxes = list(predictions)[0]['boxes'].cpu().detach().numpy()
    for box in boxes:
        x,y,m,n = box
        img = skimage.color.gray2rgb(img).astype('uint8')
        img = cv2.rectangle(img, (x,y), (m,n), (255,255,0), 1)
    return img, boxes
    
def pre_oversampling(self,imgs,labeldata):
    dend,axon,cross = [],[],[]
    dend_n,axon_n,cross_n = 0,0,0
    for num , (i,j) in enumerate(zip(labeldata[:,2],labeldata[:,3])):

        if  i.any() >0 and j.any() > 0:
            cross_n+=1
            cross.append(num)
        elif  i.any() > 0:
            dend_n += 1
            dend.append(num)
        elif  j.any() > 0:
            axon_n += 1
            axon.append(num)

    dend  = np.array(dend)
    axon  = np.array(axon)
    cross = np.array(cross)

    _,dend_multipixel = np.unique(labeldata[cross,2],return_counts=True)
    _,axon_multipixel = np.unique(labeldata[cross,3],return_counts=True)
    _,dend_pixel = np.unique(labeldata[dend,2],return_counts=True)
    _,axon_pixel = np.unique(labeldata[axon,3],return_counts=True)
    
    print(f"label variance : {np.unique(labeldata)}")
    print(f"Number of pixels : {dend_multipixel,axon_multipixel,dend_pixel,axon_pixel}")
    need_pixel = (dend_pixel[1]+dend_multipixel[1]) - (axon_pixel[1]+axon_multipixel[1])
        
    add_image = []
    add_label = []
    total_axon_pixel = 0

    # skimage.io.imsave('axontest.tif',np.swapaxes(labeldata[axon],1,3)[...,3:4].astype('uint8')*255.)
    # skimage.io.imsave('dedntest.tif',np.swapaxes(labeldata[axon],1,3)[...,2:3].astype('uint8')*255.)
    print( axon)

    while need_pixel >= total_axon_pixel:
        num_axon = np.random.choice(axon,10)
        add_axon = labeldata[num_axon]
        _,axon_pixels = np.unique(add_axon,return_counts=True)
        total_axon_pixel += axon_pixels[1]
        add_label.append(add_axon)
        add_image.append(imgs[num_axon])
    
    add_image = np.array(add_image)
    add_label = np.array(add_label)
    if add_label.ndim == 5: 
        batch,zstack,channel,img_size,_= add_label.shape
        add_image = add_image.reshape(batch*zstack,img_size,img_size)
        add_label = add_label.reshape(batch*zstack,channel,img_size,img_size)
    elif add_label.ndim == 6: 
        batch,sample,channel,zstack,img_size,_= add_label.shape
        add_image = add_image.reshape(batch*sample,zstack,img_size,img_size)
        add_label = add_label.reshape(batch*sample,channel,zstack,img_size,img_size)

    imgs = np.concatenate((imgs,add_image),axis=0)
    labels = np.concatenate((labeldata,add_label),axis=0)
    return imgs, labels