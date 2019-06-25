#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */train.py
#
# PROGRAMMER: Laura Dhatt
# DATE CREATED:     March 3, 2019                            
# REVISED DATE:     
# PURPOSE: Trains a deep neural network using a pretrained model on a dataset
#          of images, providing training loss, testing loss and model accuracy.
#          Allows the user to select one of two pretrained models, either vgg16
#          or resnet.  The image classifier is built using user input number of
#          hidden layers, and training run using user input learning rate and 
#          epochs.  This can be run on either cpu or gpu as selected by user input
# Use argparse Expected Call with <> indicating expected user input:
#      basic call:  python train.py data_directory
# some code is design from original code found on https://www.najeebhassan.com/ImageClassifierProject.html      
# Example call:
#    python train.py --dir image_path/ --arch vgg16 --hidden 3 --lr 0.05 --gpu
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

from get_input_args import get_input_args_train
from utilities import *
from model import Network

in_arg = get_input_args_train()
hidden_layers = in_arg.hidden
hidden_layers = list(map(int, hidden_layers.strip('[]').split(',')))

def testing_pass(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        if in_arg.gpu == 'gpu':
            images, labels = images.to('cuda'), labels.to('cuda')        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        #compute probabilities
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def main():
    
    #load a pre trained network (VGG16 or resnet)
    if in_arg.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.resnet(pretrained=True)
        
    num_ftrs = model.classifier[0].in_features
    print(num_ftrs)             #used as first layer of classifier
    
    #Build the model (pretrained model whose classifier is replaced with the custom classifier)
    input_size = num_ftrs
    output_size = 102           #this size would be different for a different image dataset but we assume Flowers
    classifier = Network(input_size, output_size, hidden_layers)
    
    model.classifier = classifier          #replace pretrained model classifer with custom
    print(model)                           #print the model to make sure it is correct
    
    # Freeze model parameters so we don't backpropogate through them
    for param in model.parameters():
        param.requires_grad = False
    for paramc in model.classifier.parameters():
        paramc.requires_grad = True
    
    #load the dataset from utilities.py
    train_dataset = load_data('train')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=102, shuffle=True)
    
    test_dataset = load_data('test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=102, shuffle=True)
    
    valid_dataset = load_data('valid')
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=102, shuffle=True)
    
    # image classes is a list of strings such as '17' of length 102 since there are 102 class numbers
    image_classes = train_dataset.classes
    num_classes = (len(image_classes))
  
    #set up parameters for forward pass through the network 
    print_every = 2
    steps = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), in_arg.lr)

    #FORWARD PROPOGATION
    #send images and model to CPU or GPU depending on GPU available and input argument
    cuda = torch.cuda.is_available()
    if in_arg.gpu == 'gpu' and cuda:
        device = torch.device('cuda')
        model.cuda()
        FloatTensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        model.cpu()
        FloatTensor = torch.FloatTensor
        
    for e in range(in_arg.epoch):
        running_loss = 0
        model.train()                          #make sure model is in train mode
        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()              #clear the gradients
            output = model.forward(images)     #forward pass for training
            loss = criterion(output, labels)   #compute pass loss
            loss.backward()                    #backward pass
            optimizer.step()                   #optimize
    
            running_loss += loss.item()
            #during training, run test images through model and get accuracy
            if steps % print_every == 0:
                with torch.no_grad():          #turn off gradients for validation
                    test_loss, accuracy = testing_pass(model, testloader, criterion)
    
                print('epoch: ', e,
                      'Training Loss: {:.4f}'.format(running_loss/print_every),
                      'Testing Loss: {:.4f}'.format(test_loss/len(testloader)), 
                      'Testing Accuracy: {:.4f}'.format(accuracy/len(testloader)))
                running_loss = 0
                model.train()                  #turn training mode back on for next forward train pass


                
    # Save the checkpoint 
    model.class_to_idx = train_dataset.class_to_idx
    

    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'learning_rate': in_arg.lr,
                  'batch_size': 102,
                  'classifier' : classifier,
                  'epochs': in_arg.epoch,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint,'checkpoint.pth')

    
# Call to main function to run the program
if __name__ == "__main__":
    main()