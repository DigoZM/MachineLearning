#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/model.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn

# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
DEF_NUM_FEATS = 26
NUM_NODES = 26
NUM_CLASSES = 2
NEW_LINE = "\n"

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
#
def get_data(fp, num_feats):

    # initialize the data and labels
    #
    data = []
    labels = []

    # for each line of the file
    #
    for line in fp.read().split(NEW_LINE):

        # split the string by white space
        #
        temp = line.split()

        # if we dont have 26 feats + 1 label
        #
        if not (len(temp) == num_feats + 1):
            continue

        # append the labels and data
        #
        labels.append(int(temp[0]))
        data.append([float(sample) for sample in temp[1:]])

    # exit gracefully
    #
    return data, labels
#
# end of function


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch MLP model
#
class Model(nn.Module):

    # function: init
    #
    # arguments: input_size - int representing size of input
    #            hidden_size - number of nodes in the hidden layer
    #            num_classes - number of classes to classify
    #
    # return: none
    #
    # This method is the main function.
    #
    def __init__(self, input_size, hidden_size, num_classes):

        # inherit the superclass properties/methods
        #
        super(Model, self).__init__()

        # define the model
        #
        self.neural_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes))
    #
    # end of function


    # function: forward
    #
    # arguments: data - the input to the model
    #
    # return: out - the output of the model
    #
    # This method feeds the data through the network
    #
    def forward(self, data):

        # return the output
        #
        return self.neural_net(data)
    #
    # end of method
#
# end of class

#
# end of file
