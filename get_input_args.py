#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */get_input_args_train.py
#                                                                             
# PROGRAMMER: Laura Dhatt  
# DATE CREATED: 03/02/2019                           
# REVISED DATE: 03/04/2019
# PURPOSE:  Create a function that retrieves the command line inputs
#           needed  by the training function in train.py and by the 
#           preduct function in predict.py.  If the user fails to provide 
#           some or all of the inputs, the default
#           values are used.  
  
    
import argparse
#
#
def get_input_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type = str, default = 'flowers', help = 'image dataset folder')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'vgg16 (default) or resnet only')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate, floating point')
    parser.add_argument('--hidden', type = str, default = '[8000, 4000, 1000]', help = 'hidden layers')
    parser.add_argument('--epoch', type = int, default = 1, help = 'int number of training epochs')
    parser.add_argument('--gpu', type = str, default = 'gpu', help = 'enter gpu to run on gpu')
    parser.add_argument('--save_dir', type = str, default = '/', help = 'save directory')
    
    return parser.parse_args()  
#                        
#
#                       
def get_input_args_predict():
    """
    Retrieves and parses the command line inputs 
    from the user.  If the user fails to provide inputs,
    the defaults are used.
    Command Line Arguments:
          1. image path as --imagefile with default value 'images/'
          2. K classes as --kclass with default value '5'
          3. JSON filename as --json with default value 'cat_to_name.json'
          4. predict processor type as --proc with default value 'cpu'  
    Parameters:
      None
    Returns:
      parse_args() -data structure that stores the command line arguments object
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagefile', type = str, default = 'flowers/test/17/image_03861.jpg', help = 'filename of image')
    parser.add_argument('--kclass', type = int, default = 5, help = 'number of top predictions')
    parser.add_argument('--json', type = str, default = 'cat_to_name.json', help = 'filename of class to name file for model')
    parser.add_argument('--gpu', type = str, default = 'cpu', help = 'cpu or gpu processing')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'pretrained model used for features')
    
    return parser.parse_args() 
                        
                    
