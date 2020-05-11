#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/decode.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#  python decode.py odir mfile data
#
# arguments:
#  odir: the directory where the hypotheses will be stored
#  mfile: input model file
#  data: the input data list to be decoded
#
# This script decodes data using a simple MLP model.
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch

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
NUM_ARGS = 3
SPACE = " "
HYP_EXT = ".hyp"            

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

    # ensure we have the correct number of arguments
    #
    if(len(argv) != NUM_ARGS):
        print("usage: python nedc_decode_mdl.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    # define local variables
    #
    odir = argv[0]
    mdl_path = argv[1]
    fname = argv[2]
    num_feats = DEF_NUM_FEATS
    if("DL_NUM_FEATS" in os.environ):
        num_feats = int(os.environ["DL_NUM_FEATS"])

    # if the odir doesn't exist, we make it
    #
    if not os.path.exists(odir):
        os.makedirs(odir)

    # get the hyp file name
    #
    hyp_name = os.path.splitext(os.path.basename(fname))[0] + HYP_EXT

    # set the device to use GPU if available
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get a file pointer
    #
    try:
        eval_fp = open(fname, "r")
    except (IOError, KeyError) as e:
        print("[%s]: %s" % (fname, e))
        exit(-1)

    # get array of the data
    # data: [[0, 1, ... 26], [27, 28, ...] ...]
    # labels: [0, 0, 1, ...]
    #
    eval_data, _ = get_data(eval_fp, num_feats)

    # close the file
    #
    eval_fp.close()

    # instantiate a model
    #
    model = Model(num_feats, NUM_NODES, NUM_CLASSES)

    # moves the model to the device
    #
    model.to(device)

    # set the model to evaluate
    #
    model.eval()

    # load the weights
    #
    model.load_state_dict(torch.load(mdl_path, map_location=device))

    # the output file
    #
    try:
        ofile = open(os.path.join(odir, hyp_name), 'w+')
    except IOError as e:
        print(os.path.join(odir, hyp_name))
        print("[%s]: %s" % (hyp_name, e.strerror))
        exit(-1)

    # get the number of data points
    #
    num_points = len(eval_data)

    # for each data point
    #
    for index, data_point in enumerate(eval_data):

        # print informational message
        #
        print("decoding %4d out of %d" % (index+1, num_points))    

        # pass the input through the model
        #
        output = model(torch.tensor(data_point, dtype=torch.float32).to(device))

        # write the highest probablity to the file
        #
        ofile.write(str(int(output.max(0)[1])) + SPACE 
                    + SPACE.join([str(point) for point in data_point]) + NEW_LINE)

    # close the file
    #
    ofile.close()
    
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
