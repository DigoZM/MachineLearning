# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:50:21 2020

@author: digo
"""

#!/usr/bin/env python
#
# file: $(NEDC_NFC)/util/python/nedc_eval_eeg/nedc_eval_eeg.py
#
# revision history:
#  20190909 (VK): initial version
#  20190910 (VK): commented
#
# usage:
#   python3 yinyang.py n1 n2 overlap
# 
# options:
#
# arguments:
#  n1: number of samples in Yin
#  n2: number of samplles in Ying
#  overlap: overlap of data ( a number between -1 and 1 )
#
# this script produces coordinates of points for yin-yang shapes
# and save the final scatter plot in a png file.
#------------------------------------------------------------------------------

# import system modules
#
import sys
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# number of mandatory arguments plus one (the file name included)
#
DEF_NUM_ARGS = 4

# the plot file name
#
DEF_PLOT_FILE_NAME = 'YinYangTraining.png'

# coordinates of boundary points
#
DEF_XMIN = 0
DEF_XMAX = 1
DEF_YMIN = 0
DEF_YMAX = 1

#------------------------------------------------------------------------------
#
# classes are listed here:
#  there is one class in this file.
#   setYinYang
#
#------------------------------------------------------------------------------

# class: setYinYang
#
# this class implements the main procedure to produce coordinates of YinYang
# shape in the predefined scales.
#
class setYinYang:
    
    # method: constructor
    #
    # arguments:
    #  scale: border of the shape
    #  n1: number of samples in class 1
    #  n2: number of samples in class 2
    #  overlap: the overlap parameter of two classes, a number between [-1, 1]
    #
    # return:
    #  instance of the class
    #
    def __init__(self, scale , n1 , n2 , overlap):
        
        # declare variables
        #
        self.overlap = overlap    # overlap parameter between [-1, 1]
        self.yin = None           # yin set of data
        self.yang = None          # yang set of data
        self.yin_label = 0        # label for Yin class
        self.yang_label = 1       # label for Yang class
        self.xpt = 0              # x position of a point
        self.ypt = 0              # y position of a point
        self.distance1 = 0.0      # distance of point from origin
        self.distance2 = 0.0      # distance from center of Yin
        self.distance3 = 0.0      # distance from center of Yang
        self.radius1 = 0.0        # acceptable radius for Yin
        self.radius2 = 0.0        # acceptable radius for Yang
        
        # the boundry, mean and standard deviation of plot
        #
        self.xmax = scale['xmax']
        self.xmin = scale['xmin']
        self.ymax = scale['ymax']
        self.ymin = scale['ymin']
        self.xmean = self.xmin + 0.5 * (self.xmax - self.xmin)
        self.ymean = self.ymin + 0.5 * (self.ymax - self.ymin)
        self.stddev_center = 1.5 * (self.xmax - self.xmin) / 2
         
        # creating empty lists to save points' coordinates
        #
        self.yin = []
        self.yang = []
        
        # calculate the radius of each class on the plot
        #
        self.radius1 = 1.5 * ((self.xmax - self.xmin) / 4)
        self.radius2 = 0.75 * ((self.xmax - self.xmin) / 4)
        
        # number of samples in each class
        #
        self.n_yin = n1
        self.n_yang = n2
        
        # producing some random numbers based on Normal distribution and then
        # calculating the points distance to each class, choosing the closest 
        # set.
        # the look will exit when both classes has been built up.
        #
        n_yin_counter = 0
        n_yang_counter = 0
        while ((n_yin_counter < self.n_yin) | \
                (n_yang_counter < self.n_yang)):
            
            # generate points with Normal distribution
            #
            xpt = np.random.normal(self.xmean, self.stddev_center, 1)[0]
            ypt = np.random.normal(self.ymean, self.stddev_center, 1)[0]
            
            # calculate radius for each generated point
            #
            distance1 = np.sqrt(xpt**2 + ypt**2)
            distance2 = np.sqrt(xpt**2 + (ypt + self.radius2)**2)
            distance3 = np.sqrt(xpt**2 + (ypt - self.radius2)** 2)
            
            # decide which class each point belongs to.
            # when the point added to each class, its counter increases by 1.
            # if the counter reach to the requested number of samples in 
            # each class, then no sample will appended to that class and the
            # produced point will be thrown away.
            #
            if (distance1 <= self.radius1):
                if ((xpt >= -self.radius1) & (xpt <= 0)):
                    if (((distance1 <= self.radius1) | \
                           (distance2 <= self.radius2)) & \
                           (distance3 > self.radius2)):
                        if (n_yin_counter < self.n_yin):
                            self.yin.append([xpt, ypt])
                            n_yin_counter += 1
                    elif (n_yang_counter < self.n_yang):
                        self.yang.append([xpt, ypt])
                        n_yang_counter += 1
                if ((xpt > 0.0) & (xpt <= self.radius1)):
                    if (((distance1 <= self.radius1) | \
                           (distance3 <= self.radius2)) & \
                           (distance2 > self.radius2)):
                        if (n_yang_counter < self.n_yang):
                            self.yang.append([xpt, ypt])
                            n_yang_counter += 1
                    elif (n_yin_counter < self.n_yin):
                        self.yin.append([xpt, ypt])
                        n_yin_counter += 1

        # translate each sample in Yin and Yang from the origin to
        # the center of the plot.
        # for implementing overlap, the overlap parameter multiply to one of 
        # the plot center points. So the overlap parameter interferes in 
        # translation process.
        #
        self.yang = np.array(self.yang) +\
                    np.array([self.xmean, self.ymean])
        self.yin = np.array(self.yin) +\
                   np.array([self.xmean, self.ymean])  *\
                   (1 + self.overlap)

    #
    # end of __init__

    # method: print_out
    #
    # arguments: none
    #
    # return: none
    #
    # this function print the labels and points' coordinates to the standard
    # output.
    # 
    def print_out(self):
        
        # print the label and coordinate of each point to stdout
        #
        print('The coordinates of samples are as follows:')
        for row in self.yin:
            print(self.yin_label, row[0], row[1])
        for row in self.yang:
            print(self.yang_label, row[0], row[1])
        #MINE -- write cordinates in one file
        file_object  = open('training_set.txt', 'w')
        for row in self.yin:
            file_object.write('%d %.17f %.17f\n' % (self.yin_label, row[0], row[1]))
        for row in self.yang:
            file_object.write('%d %.17f %.17f\n' % (self.yang_label, row[0], row[1]))
        file_object.close()

    #
    # end of print_out

    # method: plot
    #
    # arguments: none
    #
    # return: none
    #
    # this function save the scatter plot ot points to a png file
    #
    def plot(self):
        
        # Save the plot to a PNG file in current or specified directory.
        #
        plt.scatter(self.yang[:, 0], self.yang[:, 1], color = 'r',\
                     alpha = 0.5)
        plt.scatter(self.yin[:, 0], self.yin[:, 1], color = 'b',\
                     alpha = 0.5)
        plt.savefig(DEF_PLOT_FILE_NAME)
    
    #
    # end of plot

#
# end of setYinYang

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------
# method: main
#
# arguments:
#  argv: list of command line arguments.
#
# return: none
#
# this function is the main program.
#
def main(argv):

    # the first assumption is that the generated samples are in the range of 
    # 0 and 1.
    #
    scale = {
            'xmin': DEF_XMIN,
            'xmax': DEF_XMAX,
            'ymin': DEF_YMIN,
            'ymax': DEF_YMAX
            }

    # read the command-line arguments and check if they are inserted 
    # correctly.
    #
    if(len(sys.argv) == DEF_NUM_ARGS):
        if (sys.argv[1].isdigit() & sys.argv[2].isdigit() \
             & sys.argv[3].replace('.', '', 1).replace('-', '', 1).isdigit()):
            n1 = int(sys.argv[1])
            n2 = int(sys.argv[2])
            overlap = float(sys.argv[3])
            yy = setYinYang(scale, n1, n2, overlap)
            yy.print_out()
            yy.plot()
        else:
            print('Please enter two positive integers and one floating point number.')
    else:
        print('At least three arguments needed.')
        print('         n1: the number of samples in class 1')
        print('         n2: the number of samples in class 2')
        print('    overlap: overlap of data ( a number between -1 and 1 )')
        print('Example:')
        print('    python3 yingyang.py 1000 1000 0.1')
        print('    This generates the coordinates of 2000 samples in 2-D which')
        print('    1000 of them belongs to class 1 and 1000 of them belongs to')
        print('    class 2. They have an 0.1 overlap.')
        print('    Also it save a PNG plot of the samples too.')

#
# end of main

# begin gracefully
#
if __name__ == "__main__":
    main(sys.argv)

#
# end of file
