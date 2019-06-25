# PROGRAMMER: Laura Dhatt
# DATE CREATED:     March 3, 2019                            
# REVISED DATE:     March 4, 2019
# PURPOSE: Utilities needed by both train.py and predict.py
#          Contains function defines for load_data and process_image
#          
##

import numpy as np
import torch
import json
import torch.nn.functional as F
from torch import nn
from torch import optim

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import OrderedDict
import torchvision.models as models
from PIL import Image

def load_data(data_set):
   
    # Define your transforms for the training, validation, and testing sets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                         transforms.CenterCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), 
                         normalize])
    valid_transforms = transforms.Compose([
                         transforms.CenterCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), 
                         normalize])
    test_transforms = transforms.Compose([
                         transforms.CenterCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), 
                         normalize])

    if data_set == 'train':
        data_dir = 'flowers/train'
        rtn_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    elif data_set == 'test':
        data_dir = 'flowers/test'
        rtn_dataset = datasets.ImageFolder(data_dir, transform=test_transforms)
    elif data_set == 'valid':
        data_dir = 'flowers/valid'
        rtn_dataset = datasets.ImageFolder(data_dir, transform=valid_transforms)
   
    return rtn_dataset


#Preprocessing the IMage
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
        
    pil_image = transforms.functional.resize(image, 256)
    pil_image = pil_image.crop((16, 16, 240, 240))
       
    #change values to 0-1
    np_image = np.array(pil_image)/256
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (np_image - mean)/std
    
    return image.transpose((2, 0, 1))

