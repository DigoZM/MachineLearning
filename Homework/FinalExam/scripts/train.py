#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/train.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#  python train.py mdir data
#
# arguments:
#  mdir: the directory where the output model is stored
#  data: the input data list
#
# This script trains a simple MLP model
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam

# import the model and all of its variables/functions
#
from model import *

# import modules
#
import sys
import os

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
NUM_ARGS = 2
NUM_EPOCHS = 100
BATCH_SIZE = 36
LEARNING_RATE = "lr"
BETAS = "betas"
EPS = "eps"
WEIGHT_DECAY = "weight_decay"

# for reproducibility, we seed the rng
#
set_seed(SEED1)            

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main(argv):

    # ensure we have the correct amount of arguments
    #
    if(len(argv) != NUM_ARGS):
        print("usage: python nedc_train_mdl.py [MDL_PATH] [TRAIN_SET]")
        exit(-1)

    # define local variables
    #
    mdl_path = argv[0]
    fname = argv[1]
    num_feats = DEF_NUM_FEATS
    if("DL_NUM_FEATS" in os.environ):
        num_feats = int(os.environ["DL_NUM_FEATS"])

    # get the output directory name
    #
    odir = os.path.dirname(mdl_path)

    # if the odir doesn't exits, we make it
    #
    if not os.path.exists(odir):
        os.makedirs(odir)

    # set the device to use GPU if available
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get a file pointer
    #
    try:
        train_fp = open(fname, "r")
    except (IOError) as e:
        print("[%s]: %s" % (fname, e.strerror))
        exit(-1)

    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    train_data, train_labels = get_data(train_fp, num_feats)

    # close the file
    #
    train_fp.close()

    # instantiate a model
    #
    model = Model(num_feats, NUM_NODES, NUM_CLASSES)

    # moves the model to device (cpu in our case so no change)
    #
    model.to(device)

    # set the adam optimizer parameters
    #
    opt_params = { LEARNING_RATE: 0.005,
                   BETAS: (.9,0.999),
                   EPS: 1e-08,
                   WEIGHT_DECAY: .00001 }

    # set the loss and optimizer
    #
    loss_fx = nn.CrossEntropyLoss()
    loss_fx.to(device)

    # create an optimizer, and pass the model params to it
    #
    adam_opt = Adam(model.parameters(), **opt_params)

    # get the number of epochs to train on
    #
    epochs = NUM_EPOCHS

    # get the batch size
    #
    batch_size = BATCH_SIZE

    # get the number of batches (ceiling of train_data/batch_size)
    #
    num_batches = -(-len(train_data) // batch_size)

    # for each epoch
    #
    for epoch in range(epochs):

        # index represents the batch number
        #
        index = 0

        # for each batch in increments of batch size
        #
        for batch in range(0, len(train_data), batch_size):

            # set all gradients to 0
            #
            adam_opt.zero_grad()

            # collect the samples as a batch
            #
            batch_data = torch.tensor(train_data[batch:batch + batch_size], \
                                      dtype=torch.float32).to(device)
            batch_labels = torch.tensor(train_labels[batch:batch + batch_size]).long().to(device)

            # feed the network the batch
            #
            output = model(batch_data)

            # get the loss
            #
            loss = loss_fx(output, batch_labels)

            # perform back propagation
            #
            loss.backward()
            adam_opt.step()

            # display informational message
            #
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, index + 1, num_batches, loss.item()))

            # increment the batch number
            #
            index += 1

    # save the model
    #
    torch.save(model.state_dict(), mdl_path)

    # exit gracefully
    #
    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
