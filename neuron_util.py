import cv2
import skimage
import torch
import numpy as np
import scipy, nd2reader

from natsort import natsorted
from nd2reader import ND2Reader
from skimage.util.shape import view_as_blocks
from skimage.filters import threshold_otsu, threshold_yen

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


def multi_decode(multi_image):
    stack_img = []
    for i in range(len(multi_image)):
        s_img= decode_segmap(multi_image[i])
        stack_img.append(s_img)
    return np.array(stack_img)

def ch_channel(img):
    return  torch.argmax(img,dim=1).cpu().detach().numpy().astype('uint8')

##############preprocessing#################
def preprocessing(nd2Dir='./#12_3.nd2'):

    normalizedImg = np.zeros((1024, 1024))
    win_size = 256

    # with ND2Reader(nd2Dir) as images:
    images = ND2Reader(nd2Dir)
    nd2 = nd2reader.Nd2(nd2Dir)
    stack_images = []
    stack_mitos = []
    # load Z axis image than stack images
    with ND2Reader(nd2Dir) as images:
        images.iter_axes = 'zc'
        i = 1
        for fov in images:
            if i%2==0:
                # stack_mitocondria_images
                fov = cv2.normalize(fov,  normalizedImg, 0, 65535, cv2.NORM_MINMAX).astype('uint16')
                stack_mitos.append(fov)
            else:
                # stack_structure_images
                fov = cv2.normalize(fov,  normalizedImg, 0, 65535, cv2.NORM_MINMAX).astype('uint16')
                stack_images.append(fov)
            i+=1

        stack_images = np.array(stack_images)
        stack_mitos = np.array(stack_mitos)
        
        #projection images
        pro_img = np.max(stack_images,axis=0)
        pro_mito = np.max(stack_mitos,axis=0)

        return pro_img, pro_mito

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

def locateComponents(img,minSiz=0,maxSiz=10000):
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

    if out.ndim == 2:
        out = skimage.color.gray2rgb(out).astype('uint8')
    
    return ret, out

def detecion_box(img,predictions):
    boxes = list(predictions)[0]['boxes'].cpu().detach().numpy()
    for box in boxes:
        x,y,m,n = box
        img = skimage.color.gray2rgb(img).astype('uint8')
        img = cv2.rectangle(img, (x,y), (m,n), (255,255,0), 1)
    return img, boxes


