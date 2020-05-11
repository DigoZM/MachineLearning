#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/score.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#  python score.py ref hyp
#
# arguments:
#  ref: a file containing the reference labels
#  hyp: a matching file containing the hypotheses
#
# This script scores data.
#------------------------------------------------------------------------------

# import python modules
#
import os
import sys
import numpy as np

# main: this is the main function of this Python
#
def main(argv):

    # load the two files
    #
    refs = [line.rstrip('\n') for line in open(sys.argv[1])]
    hyps = [line.rstrip('\n') for line in open(sys.argv[2])]
    
    if len(refs) != len(hyps):
        print("error: files are not the same length")
        exit(0)

    # create a 2x2 array for scoring
    #
    conf = np.zeros((2, 2))

    # loop over the lists and check the first element
    #
    for r,h in zip(refs, hyps):
        ref = r.split()[0]
        hyp = h.split()[0]

        # count errors
        #
        conf[int(ref), int(hyp)] += int(1)

    # dump the confusion matrix
    #
    print("%0.7s %5s %5s" % ("r/h:  ", " h[0]", " h[1]"))
    print("%5s:" % ("r[0]"), "%5d" % (conf[0,0]), "%5d" % conf[0,1])
    print("%5s:" % ("r[1]"), "%5d" % (conf[1,0]), "%5d" % conf[1,1])

    # summarize the errors
    #
    print("error rate  = %10.4f%%" % ((1 - (conf[0,0] + conf[1,1])/len(refs)) * 100))

# begin gracefully
#
if __name__ == "__main__":
    main(sys.argv[0:])

#
# end of file

