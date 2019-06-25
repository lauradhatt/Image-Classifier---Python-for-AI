#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */predict.py
#
# PROGRAMMER: Laura Dhatt
# DATE CREATED:     March 3, 2019                            
# REVISED DATE:     March 4, 2019
# PURPOSE: Using a trained deep neural network, predict the classification of
#          a single image.  Runs the image through the trained network with the
#          output of topk probability and classifications passed by the user
#          
# Use argparse Expected Call with <> indicating expected user input:
#      basic call:  python preduct.py imagefile
#       
# Example call:
#    python python predict.py --arch vgg16 --imagefile flowers/valid/16/image_06671.jpg
##



import numpy as np
import torch
import json
import torch.nn.functional as F
from torch import nn
from torch import optim
from utilities import process_image

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image

from get_input_args import get_input_args_predict
in_arg = get_input_args_predict()

#Load the Checkpoint
def load_build(filepath):
    
    checkpoint = torch.load(filepath)
    
    if in_arg.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.resnet(pretrained=True)

    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    batch_size = checkpoint['batch_size']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
         
    return model


#Function to present the image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


#Class Prediction
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    #turn off dropout
    model.eval()
    topk = in_arg.kclass
    image = process_image(in_arg.imagefile)
    image = torch.from_numpy(np.array([image])).float()
    
    cuda = torch.cuda.is_available()
    if in_arg.gpu == 'gpu' and cuda:
        device = torch.device('cuda')
        model.cuda()
        FloatTensor = torch.cude.FloatTensor
    else:
        device = torch.device('cpu')
        model.cpu()
        FloatTensor = torch.FloatTensor
    
    
    #feed forward
    output = model.forward(image)
    ps = torch.exp(output)
    
    probs = torch.topk(ps, topk)[0].tolist()
    index_k = torch.topk(ps, topk)[1].tolist()
        
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    print('top {} probabilities:'.format(topk), probs)
    #print('index-k:', index_k)
    #correlate the top 5 indices with their categories
    classes = []
    for i in index_k[0]:
        classes.append(idx_to_class[i])
    
    return probs, classes

def main():
    
    #rebuild the model
    model = load_build('checkpoint.pth') 
    
    probs, classes = predict(in_arg.imagefile, model, in_arg.kclass)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    

    #SANITY CHECKING
    # Display top 5 classes and probabilities
    max_index = np.argmax(probs)
    label = classes[max_index]
    labels = []
    for i in classes:
        labels.append(cat_to_name[i])
    print('top {} classes:  '.format(in_arg.kclass), classes)
    print('top {} categories:  '.format(in_arg.kclass), labels)    
    
    #fig = plt.figure(figsize=(6, 10))
    #ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
    #ax2 = plt.subplot2grid((14,10), (9,2), colspan=6, rowspan=9)

    #image = Image.open(image_path)

    #ax1.axis('off')
    #ax1.set_title(cat_to_name[label])
    #ax1.imshow(image)
    
    #y_pos = [0, 1, 2, 3, 4]
    #ax2.set_yticks(y_pos)
    #ax2.set_yticklabels(labels)
    #ax2.set_xlabel('Probability')
    #ax2.invert_yaxis()
    #ax2.barh(y_pos, probs[0], xerr=0, align='center', color='blue')
    
    #plt.show()

# Call to main function to run the program
if __name__ == "__main__":
    main()


