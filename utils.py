import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2) 
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1) 
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach())
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)
    return loss1 + loss2

def P_loss(im1, X1):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss

def joint_RGB_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('RGB',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))      
    return result

def joint_L_horizontal(im1, im2):
    if im1.size==im2.size:
        w, h = im1.size
        result = Image.new('L',(w*2, h))
        result.paste(im1, box=(0,0))
        result.paste(im2, box=(w,0))   
    return result
