#!/usr/bin/env python

# this script scores your recognition results. run it like this:
#
#  python score.py ref.txt hyp.txt
#
# the first file is the reference; the second file is the hypothesis.
#

# import required modules:
#  note that the path to the module htkmfc must be included in the
#  PYTHONPATH environment variable.
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
        print "error: files are not the same length"
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
    print "%4s %5s %5s" % ("", "ref", "hyp")
    print "%3s:" % ("ref"), "%5d" % (conf[0,0]), "%5d" % conf[0,1]
    print "%3s:" % ("hyp"), "%5d" % (conf[1,0]), "%5d" % conf[1,1]

    # summarize the errors
    #
    print "error rate  = %10.4f%%" % ((1 - (conf[0,0] + conf[1,1])/len(refs)) * 100)

# begin gracefully
#
if __name__ == "__main__":
    main(sys.argv[0:])

#
# end of file
